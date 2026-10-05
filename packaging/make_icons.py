"""Draw the Drawing2FEA app icon (PNG, ICO, ICNS) with Pillow.

    python packaging/make_icons.py

Motif: a blueprint sheet (2D drawing) behind an isometric L-bracket whose
faces are shaded like a stress contour (3D model + analysis).
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

S = 1024
OUT = Path(__file__).resolve().parent.parent / "drawing2fea" / "gui" / "assets"


def lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(len(a)))


def draw() -> Image.Image:
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    # rounded-square background with a vertical gradient
    bg = Image.new("RGBA", (S, S))
    top, bot = (32, 92, 196), (14, 44, 110)
    px = bg.load()
    for y in range(S):
        c = lerp(top, bot, y / (S - 1)) + (255,)
        for x in range(S):
            px[x, y] = c
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).rounded_rectangle((40, 40, S - 40, S - 40), radius=210, fill=255)
    img.paste(bg, (0, 0), mask)
    d = ImageDraw.Draw(img)

    # blueprint sheet (2D drawing), slightly rotated look via offset rectangle
    sheet = (150, 150, 640, 700)
    shadow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle((sheet[0] + 14, sheet[1] + 18, sheet[2] + 14, sheet[3] + 18),
                                             radius=26, fill=(0, 0, 0, 90))
    img.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(14)))
    d.rounded_rectangle(sheet, radius=26, fill=(236, 243, 255, 255))
    ink = (40, 96, 190, 255)
    # front view + top view + side view of the bracket on the sheet
    d.rectangle((200, 470, 420, 640), outline=ink, width=10)        # front
    d.line((200, 590, 420, 590), fill=ink, width=8)
    d.ellipse((285, 495, 335, 545), outline=ink, width=8)
    d.rectangle((200, 220, 420, 420), outline=ink, width=10)        # top
    d.line((200, 270, 420, 270), fill=ink, width=8)
    d.rectangle((460, 470, 600, 640), outline=ink, width=10)        # side
    for y0 in range(470, 640, 34):                                 # hidden line (dashed)
        d.line((535, y0, 535, y0 + 18), fill=ink, width=6)

    # isometric L-bracket in front, shaded like a stress contour
    def iso(x, y, z, ox=470, oy=700, k=1.15):
        # orthographic view from front-right-top (view direction (1, -1, 1))
        return (ox + (x + y) * 0.707 * k, oy + (x - y - 2 * z) * 0.408 * k)

    cold, warm, hot = (255, 214, 102), (247, 140, 52), (214, 48, 39)
    W, D, T, Hh = 330, 200, 55, 270   # width, depth, base thickness, wall height (wall at back y in [D-60, D])
    faces = [
        # base top
        ([iso(0, 0, T), iso(W, 0, T), iso(W, D - 60, T), iso(0, D - 60, T)], cold),
        # base front
        ([iso(0, 0, 0), iso(W, 0, 0), iso(W, 0, T), iso(0, 0, T)], warm),
        # base right side
        ([iso(W, 0, 0), iso(W, D, 0), iso(W, D, T), iso(W, 0, T)], lerp(warm, hot, 0.4)),
        # wall front
        ([iso(0, D - 60, T), iso(W, D - 60, T), iso(W, D - 60, Hh), iso(0, D - 60, Hh)], lerp(cold, warm, 0.55)),
        # wall right
        ([iso(W, D - 60, T), iso(W, D, T), iso(W, D, Hh), iso(W, D - 60, Hh)], hot),
        # wall top
        ([iso(0, D - 60, Hh), iso(W, D - 60, Hh), iso(W, D, Hh), iso(0, D, Hh)], (255, 236, 170)),
    ]
    sh = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    sd = ImageDraw.Draw(sh)
    for poly, _ in faces:
        sd.polygon([(x + 16, y + 22) for x, y in poly], fill=(0, 0, 0, 110))
    img.alpha_composite(sh.filter(ImageFilter.GaussianBlur(16)))
    d = ImageDraw.Draw(img)
    edge = (90, 30, 20, 255)
    for poly, col in faces:
        d.polygon(poly, fill=col + (255,))
        d.line(poly + [poly[0]], fill=edge, width=7, joint="curve")
    # hole on the wall front (a circle in the x-z plane, projected)
    import math

    zc = (T + Hh) / 2 + 15
    ring = [iso(W / 2 + 48 * math.cos(a), D - 60, zc + 48 * math.sin(a))
            for a in [2 * math.pi * k / 48 for k in range(48)]]
    d.polygon(ring, fill=(120, 30, 25, 255))
    d.line(ring + [ring[0]], fill=edge, width=6)
    return img


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    big = draw()
    big.resize((512, 512), Image.LANCZOS).save(OUT / "icon.png")
    big.resize((256, 256), Image.LANCZOS).save(OUT / "icon.ico",
                                               sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64),
                                                      (128, 128), (256, 256)])
    big.save(OUT / "icon.icns")
    print("written to", OUT)


if __name__ == "__main__":
    main()
