# drawing2fea — 2D 도면 → 3D 모델링 → 구조해석 자동화

2D 도면(DXF 또는 JSON)만 입력하면 **3D 솔리드 모델 생성, 유한요소 메쉬, 정적 구조해석(응력·변위·안전율), 고유진동수 해석**을 한 번에 자동으로 수행하고 STL / VTK / HTML 보고서를 생성하는 Python 프로그램입니다.

```
2D 도면 (.dxf/.json)
  └─ ① 도면 해석: LINE/ARC/CIRCLE/LWPOLYLINE(bulge)/SPLINE 연결 → 닫힌 윤곽 + 구멍 자동 인식
       └─ ② 3D 모델링: 돌출(extrude) / 회전(revolve) → 사면체 메쉬 (tet10 2차 요소)
            └─ ③ 해석: 정적(von Mises, 주응력, 변위, 반력, 안전율) + 모달(고유진동수·모드형상)
                 └─ ④ 출력: model.stl, results.vtk(ParaView), results.json, report.html
```

## 설치

```bash
pip install -r requirements.txt          # numpy, scipy, ezdxf, matplotlib
pip install pypardiso pyamg               # (선택) 고속 솔버 — 대형 모델에서 10배 이상 빠름
```

## 빠른 시작

```bash
python -m drawing2fea examples                    # 예제 도면 생성 (examples/)
python -m drawing2fea run examples/bracket.dxf    # 전체 자동 수행 → results/
python -m drawing2fea run examples/flange.dxf -o results_flange
python -m drawing2fea run examples/bracket.dxf -c examples/job_bracket.json   # 설정 파일 사용
python -m drawing2fea materials                   # 재료 라이브러리
```

출력 예 (`examples/bracket.dxf`, AL6061 브래킷, 볼트 구멍 2개 고정, 하중 구멍에 3 kN):

```
[1/5] drawing: 1 region(s), 4 hole(s), area 6891.67 mm²
[2/5] 3D model: extrude, volume 68977.5 mm³, 8793 tet10 / 15106 nodes
[3/5] static: max σ_vm 119.6 MPa, max |u| 0.3796 mm, SF 2.31
[4/5] modal: 473.4, 2073, 2420, 2836 Hz
[5/5] outputs written to results/
```

## 도면 작성 규칙

* **윤곽**: 닫힌 폴리라인/원, 또는 끝점이 이어진 LINE·ARC·SPLINE 조각(자동 연결). 가장 바깥 윤곽이 외곽, 그 안의 닫힌 윤곽은 구멍으로 자동 분류됩니다 (구멍 안의 섬도 지원).
* 레이어 이름에 `DIM, CENTER, HIDDEN, ANNO, TEXT, AXIS, HATCH` 가 포함된 도형은 무시합니다 (`--layers PROFILE` 로 특정 레이어만 읽기 가능).
* 구멍 번호는 **왼쪽→오른쪽, 아래→위** 순서 (`hole:0`, `hole:1`, …). 보고서 도면 그림에 번호가 표시됩니다.
* 단위: mm, N, MPa, t(톤), s.

### 도면 안에 해석 조건 쓰기 (TEXT / MTEXT, `KEY=VALUE`)

도면에 아래 문자를 넣으면 **도면 하나만으로** 해석이 완전히 자동 수행됩니다.

| 키 | 의미 | 예 |
|---|---|---|
| `THICKNESS` / `THK` | 돌출 두께 [mm] | `THICKNESS=10` |
| `REVOLVE` / `ANGLE` | 회전 각도 [deg], 축 x = `AXIS_X` (3D z축) | `REVOLVE=360` |
| `MATERIAL` / `MAT` | 재료 | `MATERIAL=AL6061` |
| `MESH` | 목표 요소 크기 [mm] | `MESH=4` |
| `ORDER` | 요소 차수 1(tet4) / 2(tet10, 기본) | `ORDER=2` |
| `FIX` | 완전 고정 면 (`;` 로 여러 개) | `FIX=hole:0;hole:1` |
| `FORCE` | 면에 분포되는 총 하중 [N] | `FORCE=hole:3:0,-3000,0` |
| `PRESSURE` | 압력 [MPa], +는 면을 누름 | `PRESSURE=r<=12:30` |
| `GRAVITY` | 가속도 [mm/s²] | `GRAVITY=0,-9810,0` |
| `MODES` | 고유진동수 개수 (0=생략) | `MODES=6` |

경계조건/하중이 없으면 `xmin`(돌출) 또는 `zmin`(회전) 면 고정 + 자중을 가정하고 보고서에 명시합니다.

### 면 선택자 (selector)

| 선택자 | 의미 |
|---|---|
| `xmin`, `xmax`, `ymin`, `ymax`, `zmin`/`bottom`, `zmax`/`top` | 바운딩 박스 평면 위의 면 |
| `hole:<k>`, `hole:all`, `outer` | 구멍 k의 내벽 / 모든 구멍 / 외곽 측면 (돌출 전용) |
| `x<10`, `y>=5`, `z=0`, `r<=12.5` | 좌표 조건 (r = √(x²+y²), 회전체의 보어 등) |
| `box:x0,x1,y0,y1,z0,z1` | 박스 영역 (`*` = 무한) |
| `A & B` | 교집합, 예: `xmin & y>20` |

## 설정 파일 (JSON/YAML)

우선순위: 기본값 < 도면 주석 < 설정 파일 < 명령행 옵션. 예: [`examples/job_bracket.json`](examples/job_bracket.json)

```json
{
  "model": {"operation": "extrude", "thickness": 10},
  "material": "AL6061",
  "mesh": {"size": 3, "order": 2, "max_elements": 20000},
  "boundary_conditions": [{"type": "fixed", "on": "hole:0"},
                          {"type": "displacement", "on": "xmin", "x": 0.0},
                          {"type": "symmetry", "on": "zmin", "normal": "z"}],
  "loads": [{"type": "force", "on": "hole:3", "value": [0, -3000, 0]},
            {"type": "pressure", "on": "ymax", "value": 1.5},
            {"type": "gravity", "value": [0, -9810, 0]}],
  "analysis": {"static": true, "modal": 6},
  "solver": "auto"
}
```

재료는 라이브러리 이름(`STEEL, SS304, AL6061, TI6AL4V, COPPER, ABS`) 또는 `{"name": "MY", "E": 200000, "nu": 0.3, "rho": 7.8e-9, "yield_strength": 350}`.

## 출력 파일

| 파일 | 내용 |
|---|---|
| `report.html` | 단일 파일 보고서: 요약 지표, 도면/메쉬, 3D 모델, 응력·변위 컨투어, 모드 형상 |
| `model.stl` | 3D 모델 표면 (CAD/3D 프린팅/뷰어) |
| `results.vtk` | 메쉬 + 변위, von Mises, 응력텐서, 주응력, 모드형상 (ParaView에서 열기) |
| `results.json` | 모든 수치 결과와 실제 사용된 설정 |

## 검증 (tests/)

`python -m pytest` — 17개 테스트, 해석해와 비교:

| 항목 | 기준 | 결과 |
|---|---|---|
| 외팔보 끝단 처짐 (tet10) | Euler–Bernoulli FL³/3EI | 오차 < 0.1 % |
| 외팔보 1차 고유진동수 | 1.875² /2π · √(EI/ρAL⁴) | 오차 ≈ 0.2 % |
| 균일 인장 patch test (tet4/tet10) | σ = F/A | 1e-8 이내 일치 |
| 굽힘 응력, 반력 평형, 자중·압력 합력 | 해석해 | 일치 |
| 돌출/회전 체적, 질량·관성 | 해석해 | 일치 |
| 구속 부족 모델 | 강체운동 검출 | 오류 메시지 |

## 구조

```
drawing2fea/
  drawing.py   DXF/JSON 읽기, 조각 연결, 외곽/구멍 분류, 주석 파싱
  mesh2d.py    구멍 포함 2D 삼각분할 (경계 재샘플링 + Delaunay + 스무딩)
  model3d.py   돌출/회전 → 프리즘 → 정합 사면체 분할, tet10 변환
  fem.py       강성/질량 행렬, 정적·모달 솔버(SuperLU / PARDISO / AMG), 응력 복원
  bc.py        면 선택자, 경계조건, 하중
  config.py    기본값·도면 주석·설정 파일 병합
  export.py    STL, VTK
  report.py    그림 + HTML 보고서
  pipeline.py  전체 자동화
  cli.py       명령행
```

## 한계

* 선형 탄성, 미소 변형, 정적/모달 해석. 접촉·소성·좌굴은 미지원.
* 3D 형상은 단일 단면의 돌출 또는 회전으로 만들 수 있는 부품에 한정 (다중 뷰 도면 재구성은 미지원).
* 래스터 이미지(스캔 도면)는 DXF로 변환 후 사용.
* 고정 경계 모서리 등 기하학적 특이점의 응력 피크는 메쉬를 세분할수록 커질 수 있으므로 해석 시 주의.
