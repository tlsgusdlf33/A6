"""Linear elastic finite element solver for tet4 / tet10 meshes.

* consistent stiffness and mass matrices (vectorised numpy assembly)
* static analysis with fixed / prescribed displacements, surface forces,
  pressure and gravity
* modal analysis (natural frequencies) by shift-invert Lanczos
* nodal stress recovery (averaged), von Mises stress
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.special import roots_jacobi

from .materials import Material
from .model3d import Mesh3D

CHUNK = 4000  # elements per assembly batch (bounds peak memory)


# --------------------------------------------------------------------------- #
# reference element
# --------------------------------------------------------------------------- #
def tet_quadrature(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Conical-product Gauss rule on the unit tet, exact to degree 2n-1.

    Collapses the cube onto the tet (Duffy) and absorbs the Jacobian with
    Gauss-Jacobi weights, so all weights are positive. Weights sum to 1/6.
    """
    def gj(alpha):
        x, w = roots_jacobi(n, alpha, 0.0)
        return (x + 1) / 2, w / 2 ** (alpha + 1)

    u, wu = gj(2.0)
    v, wv = gj(1.0)
    w, ww = gj(0.0)
    U, V, W = np.meshgrid(u, v, w, indexing="ij")
    WU, WV, WW = np.meshgrid(wu, wv, ww, indexing="ij")
    xi = U
    eta = V * (1 - U)
    zeta = W * (1 - U) * (1 - V)
    pts = np.c_[xi.ravel(), eta.ravel(), zeta.ravel()]
    wts = (WU * WV * WW).ravel()
    return pts, wts


def shape_functions(order: int, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shape functions N (q, nn) and derivatives dN/dxi (q, nn, 3)."""
    pts = np.atleast_2d(pts)
    xi, eta, zeta = pts.T
    L = np.stack([1 - xi - eta - zeta, xi, eta, zeta], axis=1)  # (q,4)
    # dL/dxi (4,3)
    dL = np.array([[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    q = len(pts)
    if order == 1:
        return L, np.broadcast_to(dL, (q, 4, 3)).copy()
    from .model3d import TET10_EDGES

    N = np.empty((q, 10))
    dN = np.empty((q, 10, 3))
    N[:, :4] = L * (2 * L - 1)
    dN[:, :4, :] = (4 * L - 1)[:, :, None] * dL[None, :, :]
    for e, (a, b) in enumerate(TET10_EDGES):
        N[:, 4 + e] = 4 * L[:, a] * L[:, b]
        dN[:, 4 + e, :] = 4 * (L[:, a, None] * dL[b] + L[:, b, None] * dL[a])
    return N, dN


def node_natural_coords(order: int) -> np.ndarray:
    from .model3d import TET10_EDGES

    c = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    if order == 1:
        return c
    return np.vstack([c, 0.5 * (c[TET10_EDGES[:, 0]] + c[TET10_EDGES[:, 1]])])


def elasticity_matrix(E: float, nu: float) -> np.ndarray:
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[np.arange(3), np.arange(3)] = lam + 2 * mu
    D[np.arange(3, 6), np.arange(3, 6)] = mu
    return D


def _jacobian(X: np.ndarray, dN: np.ndarray):
    """X (ne,nn,3), dN (nn,3) -> J (ne,3,3) [dx_i/dxi_j], detJ, dN/dx (ne,nn,3)."""
    J = np.einsum("eni,nj->eij", X, dN)
    det = np.linalg.det(J)
    invJ = np.linalg.inv(J)
    dNdx = np.einsum("nj,eji->eni", dN, invJ)
    return J, det, dNdx


def _B_matrix(dNdx: np.ndarray) -> np.ndarray:
    ne, nn, _ = dNdx.shape
    B = np.zeros((ne, 6, 3 * nn))
    dx, dy, dz = dNdx[..., 0], dNdx[..., 1], dNdx[..., 2]
    B[:, 0, 0::3] = dx
    B[:, 1, 1::3] = dy
    B[:, 2, 2::3] = dz
    B[:, 3, 0::3] = dy
    B[:, 3, 1::3] = dx
    B[:, 4, 1::3] = dz
    B[:, 4, 2::3] = dy
    B[:, 5, 0::3] = dz
    B[:, 5, 2::3] = dx
    return B


def _element_dofs(el: np.ndarray) -> np.ndarray:
    return (3 * el[:, :, None] + np.arange(3)[None, None, :]).reshape(len(el), -1)


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #
def assemble_stiffness(mesh: Mesh3D, mat: Material) -> sp.csr_matrix:
    D = elasticity_matrix(mat.E, mat.nu)
    pts, wts = tet_quadrature(1 if mesh.order == 1 else 2)
    _, dN = shape_functions(mesh.order, pts)
    ndof = 3 * len(mesh.nodes)
    K = sp.csr_matrix((ndof, ndof))
    for s in range(0, len(mesh.elements), CHUNK):
        el = mesh.elements[s:s + CHUNK]
        X = mesh.nodes[el]
        Ke = 0.0
        for q, w in enumerate(wts):
            _, det, dNdx = _jacobian(X, dN[q])
            if (det <= 0).any():
                raise ValueError("inverted element encountered during assembly")
            B = _B_matrix(dNdx)
            DB = np.matmul(D, B) * (det * w)[:, None, None]
            Ke = Ke + np.matmul(B.transpose(0, 2, 1), DB)
        dofs = _element_dofs(el)
        nd = dofs.shape[1]
        rows = np.repeat(dofs, nd, axis=1).ravel()
        cols = np.tile(dofs, (1, nd)).ravel()
        K = K + sp.csr_matrix((Ke.ravel(), (rows, cols)), shape=(ndof, ndof))
    return K


def assemble_mass(mesh: Mesh3D, mat: Material) -> sp.csr_matrix:
    pts, wts = tet_quadrature(2 if mesh.order == 1 else 3)
    N, dN = shape_functions(mesh.order, pts)
    nn = N.shape[1]
    ndof = 3 * len(mesh.nodes)
    M = sp.csr_matrix((ndof, ndof))
    for s in range(0, len(mesh.elements), CHUNK):
        el = mesh.elements[s:s + CHUNK]
        X = mesh.nodes[el]
        Me = np.zeros((len(el), nn, nn))
        for q, w in enumerate(wts):
            _, det, _ = _jacobian(X, dN[q])
            Me += np.outer(N[q], N[q])[None] * (mat.rho * det * w)[:, None, None]
        # expand scalar mass to 3 dof per node
        rows_n = np.repeat(el, nn, axis=1)
        cols_n = np.tile(el, (1, nn))
        vals = Me.reshape(len(el), -1)
        rows = (3 * rows_n[..., None] + np.arange(3)).ravel()
        cols = (3 * cols_n[..., None] + np.arange(3)).ravel()
        vals = np.repeat(vals[..., None], 3, axis=2).ravel()
        M = M + sp.csr_matrix((vals, (rows, cols)), shape=(ndof, ndof))
    return M


def body_force_vector(mesh: Mesh3D, mat: Material, accel: np.ndarray) -> np.ndarray:
    """Consistent nodal loads for a uniform acceleration field (gravity)."""
    pts, wts = tet_quadrature(2)
    N, dN = shape_functions(mesh.order, pts)
    nodal = np.zeros(len(mesh.nodes))
    X = mesh.nodes[mesh.elements]
    for q, w in enumerate(wts):
        _, det, _ = _jacobian(X, dN[q])
        np.add.at(nodal, mesh.elements, N[q][None, :] * (det * w)[:, None])
    return (mat.rho * nodal[:, None] * np.asarray(accel, float)[None, :]).ravel()


def face_load_vector(mesh: Mesh3D, faces_full: np.ndarray, traction: np.ndarray) -> np.ndarray:
    """Consistent nodal loads for uniform traction on each face.

    ``traction`` is (k,3) per face or (3,). Linear tri: A/3 per corner.
    Quadratic tri6: 0 per corner and A/3 per midside node.
    """
    f = np.zeros((len(mesh.nodes), 3))
    p = mesh.nodes[faces_full[:, :3]]
    area = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
    t = np.broadcast_to(np.asarray(traction, float), (len(faces_full), 3))
    share = (area / 3.0)[:, None] * t
    cols = faces_full[:, 3:6] if faces_full.shape[1] == 6 else faces_full[:, :3]
    for k in range(3):
        np.add.at(f, cols[:, k], share)
    return f.ravel()


def face_normals_areas(mesh: Mesh3D, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = mesh.nodes[faces[:, :3]]
    n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    a = np.linalg.norm(n, axis=1)
    return n / a[:, None], 0.5 * a


# --------------------------------------------------------------------------- #
# solvers
# --------------------------------------------------------------------------- #
def _rigid_body_modes(nodes: np.ndarray) -> np.ndarray:
    n = len(nodes)
    B = np.zeros((3 * n, 6))
    c = nodes - nodes.mean(axis=0)
    for d in range(3):
        B[d::3, d] = 1.0
    x, y, z = c.T
    B[0::3, 3], B[1::3, 3] = -y, x
    B[1::3, 4], B[2::3, 4] = -z, y
    B[0::3, 5], B[2::3, 5] = z, -x
    return B


DIRECT_LIMIT = 40_000         # static: free DOFs above which AMG-CG replaces SuperLU
MODAL_DIRECT_LIMIT = 150_000  # modal: one factorisation is reused by every Lanczos step


_PARDISO_STATE: dict = {}


def _find_mkl_rt() -> str | None:
    import glob
    import site
    import sys

    roots = {sys.prefix, sys.base_prefix, "/usr/local", "/usr"}
    roots.update(site.getsitepackages() if hasattr(site, "getsitepackages") else [])
    pats = ["lib/libmkl_rt.so*", "lib/libmkl_rt*.dylib", "Library/bin/mkl_rt*.dll", "../../libmkl_rt.so*"]
    for root in roots:
        for pat in pats:
            hits = sorted(glob.glob(os.path.join(root, pat)))
            if hits:
                return os.path.abspath(hits[0])
    return None


def _pardiso():
    """Return the pypardiso module if it (and MKL) is usable, else None.

    PARDISO is an optional accelerator (``pip install pypardiso``): a
    multithreaded sparse direct solver that is typically 10x faster than
    SciPy's SuperLU on 3D elasticity problems.
    """
    if "mod" in _PARDISO_STATE:
        return _PARDISO_STATE["mod"]
    mod = None
    if os.environ.get("DRAWING2FEA_NO_PARDISO") != "1":
        try:
            if "PYPARDISO_MKL_RT" not in os.environ:
                path = _find_mkl_rt()
                if path:
                    os.environ["PYPARDISO_MKL_RT"] = path
            import pypardiso

            mod = pypardiso
        except Exception:  # ImportError, or MKL runtime missing
            mod = None
    _PARDISO_STATE["mod"] = mod
    return mod


def factorize(A: sp.spmatrix):
    """Return (solve(b) callable, description) for a sparse SPD matrix."""
    pp = _pardiso()
    if pp is not None:
        solver = pp.PyPardisoSolver()
        A = A.tocsr()
        solver.factorize(A)
        return (lambda b: solver.solve(A, b)), "direct (PARDISO)"
    lu = spla.splu(A.tocsc())
    return lu.solve, "direct (SuperLU)"


def _amg(A: sp.spmatrix, near_null: np.ndarray | None):
    import pyamg

    return pyamg.smoothed_aggregation_solver(A.tocsr(), B=near_null, symmetry="symmetric")


def solve_linear(K: sp.csr_matrix, f: np.ndarray, near_null: np.ndarray | None = None,
                 method: str = "auto") -> tuple[np.ndarray, str]:
    n = K.shape[0]
    if method == "auto":
        method = "direct" if n <= DIRECT_LIMIT or _pardiso() is not None else "amg"
    if method == "amg":
        try:
            ml = _amg(K, near_null)
        except ImportError:
            method = "direct"
        else:
            res: list[float] = []
            x = ml.solve(f, tol=1e-10, accel="cg", maxiter=2000, residuals=res)
            return x, f"AMG-preconditioned CG ({len(res) - 1} iterations)"
    if method == "direct":
        solve, desc = factorize(K)
        return solve(f), desc
    raise ValueError(f"unknown solver '{method}'")


@dataclass
class Constraints:
    dofs: np.ndarray       # constrained global dof indices
    values: np.ndarray     # prescribed values


def check_constraints(nodes: np.ndarray, cons: Constraints) -> None:
    """Raise if the constraints leave a rigid body motion free (singular stiffness)."""
    rbm = _rigid_body_modes(nodes)
    size = float(np.ptp(nodes, axis=0).max()) or 1.0
    rbm[:, 3:] /= size
    sv = np.linalg.svd(rbm[cons.dofs], compute_uv=False) if len(cons.dofs) else np.zeros(1)
    rank = int((sv > 1e-8 * max(sv.max(), 1e-300)).sum()) if len(cons.dofs) else 0
    if rank < 6:
        raise RuntimeError(
            f"the model is under-constrained: {6 - rank} rigid body motion(s) are not restrained. "
            "Add boundary conditions (e.g. FIX=xmin)."
        )


def solve_static(mesh: Mesh3D, K: sp.csr_matrix, f: np.ndarray, cons: Constraints, method: str = "auto") -> tuple[np.ndarray, np.ndarray, str]:
    """Return (u, reactions, solver description)."""
    ndof = K.shape[0]
    if len(cons.dofs) == 0:
        raise ValueError("static analysis needs at least one displacement constraint")
    check_constraints(mesh.nodes, cons)
    u = np.zeros(ndof)
    u[cons.dofs] = cons.values
    free = np.setdiff1d(np.arange(ndof), cons.dofs)
    Kff = K[free][:, free]
    rhs = f[free] - K[free][:, cons.dofs] @ cons.values
    with warnings.catch_warnings():
        warnings.simplefilter("error", spla.MatrixRankWarning)
        try:
            near_null = _rigid_body_modes(mesh.nodes)[free]
            uf, desc = solve_linear(Kff.tocsr(), rhs, near_null, method)
        except (spla.MatrixRankWarning, RuntimeError) as exc:
            raise RuntimeError("stiffness matrix is singular: the model is under-constrained (rigid body motion)") from exc
    if not np.all(np.isfinite(uf)):
        raise RuntimeError("solution is not finite: the model is probably under-constrained")
    u[free] = uf
    reactions = K @ u - f
    return u, reactions, desc


def solve_modal(K: sp.csr_matrix, M: sp.csr_matrix, cons: Constraints, n_modes: int = 6,
                nodes: np.ndarray | None = None, method: str = "auto") -> tuple[np.ndarray, np.ndarray, str]:
    """Return natural frequencies [Hz], mode shapes (ndof, n_modes) and the method used.

    Shift-invert Lanczos (ARPACK). The inverse operator is a sparse direct
    factorisation, or for large models without PARDISO an AMG-preconditioned
    CG solve, which needs far less memory than factorising a 3D stiffness matrix.
    """
    ndof = K.shape[0]
    free = np.setdiff1d(np.arange(ndof), cons.dofs)
    Kff = K[free][:, free].tocsr()
    Mff = M[free][:, free].tocsr()
    k = min(n_modes, len(free) - 2)
    # A shift keeps the operator non-singular for unconstrained (free-free) models.
    scale = Kff.diagonal().mean() / Mff.diagonal().mean()
    shift = 1e-6 * scale if len(cons.dofs) == 0 else 0.0
    A = (Kff + shift * Mff).tocsr() if shift else Kff

    if method == "auto":
        method = "direct" if len(free) <= MODAL_DIRECT_LIMIT or _pardiso() is not None else "amg"
    if method == "amg":
        try:
            near_null = _rigid_body_modes(nodes)[free] if nodes is not None else None
            ml = _amg(A, near_null)
        except ImportError:
            method = "direct"
    if method == "amg":
        # Shift-invert with an inner AMG-CG solve: memory stays O(nnz).
        def solve(b):
            return ml.solve(b, tol=1e-10, accel="cg", maxiter=2000)
        desc = "AMG-CG"
    else:
        solve, desc = factorize(A)
    OPinv = spla.LinearOperator(A.shape, matvec=solve, dtype=float)
    vals, vecs = spla.eigsh(A, k=k, M=Mff, sigma=0.0, which="LM", OPinv=OPinv)
    desc = f"shift-invert Lanczos, {desc}"
    order = np.argsort(vals)[:k]
    vals, vecs = vals[order] - shift, vecs[:, order]
    freqs = np.sqrt(np.clip(vals, 0.0, None)) / (2 * np.pi)
    modes = np.zeros((ndof, k))
    modes[free] = vecs
    return freqs, modes, desc


# --------------------------------------------------------------------------- #
# post-processing
# --------------------------------------------------------------------------- #
def nodal_stress(mesh: Mesh3D, mat: Material, u: np.ndarray) -> np.ndarray:
    """Element stresses evaluated at element nodes, averaged at shared nodes.

    Returns (n_nodes, 6) in order [sxx, syy, szz, sxy, syz, szx] (MPa).
    """
    D = elasticity_matrix(mat.E, mat.nu)
    nat = node_natural_coords(mesh.order)
    _, dN = shape_functions(mesh.order, nat)
    acc = np.zeros((len(mesh.nodes), 6))
    cnt = np.zeros(len(mesh.nodes))
    for s in range(0, len(mesh.elements), CHUNK):
        el = mesh.elements[s:s + CHUNK]
        X = mesh.nodes[el]
        ue = u[_element_dofs(el)]
        for a in range(len(nat)):
            _, _, dNdx = _jacobian(X, dN[a])
            sig = np.matmul(_B_matrix(dNdx), ue[:, :, None])[:, :, 0] @ D.T
            np.add.at(acc, el[:, a], sig)
            np.add.at(cnt, el[:, a], 1.0)
    return acc / np.maximum(cnt, 1)[:, None]


def von_mises(s: np.ndarray) -> np.ndarray:
    sxx, syy, szz, sxy, syz, szx = s.T
    return np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3 * (sxy**2 + syz**2 + szx**2))


def principal_stresses(s: np.ndarray) -> np.ndarray:
    T = np.empty((len(s), 3, 3))
    T[:, 0, 0], T[:, 1, 1], T[:, 2, 2] = s[:, 0], s[:, 1], s[:, 2]
    T[:, 0, 1] = T[:, 1, 0] = s[:, 3]
    T[:, 1, 2] = T[:, 2, 1] = s[:, 4]
    T[:, 0, 2] = T[:, 2, 0] = s[:, 5]
    return np.linalg.eigvalsh(T)[:, ::-1]


def mass_properties(mesh: Mesh3D, mat: Material) -> dict:
    """Volume, mass, centroid and inertia tensor about the centroid (corner-based, exact for straight tets)."""
    p = mesh.nodes[mesh.corners]
    vol = mesh.element_volumes()
    V = vol.sum()
    cen = (p.mean(axis=1) * vol[:, None]).sum(axis=0) / V
    # second moments: integral of x_i x_j over a tet = V/20 * (sum_a x_ai x_aj + sum x_i sum x_j)
    q = p - cen
    s = q.sum(axis=1)
    second = (np.einsum("eai,eaj->eij", q, q) + np.einsum("ei,ej->eij", s, s)) * (vol / 20.0)[:, None, None]
    S = second.sum(axis=0) * mat.rho
    I = np.trace(S) * np.eye(3) - S
    return {
        "volume_mm3": float(V),
        "mass_kg": float(V * mat.rho * 1000.0),
        "centroid_mm": cen.tolist(),
        "inertia_t_mm2": I.tolist(),
    }
