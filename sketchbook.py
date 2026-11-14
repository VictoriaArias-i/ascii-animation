"""Dark sketchbook page with hand-drawn style botanical flowers (all drawn in code).

page = build_page(w, h)  ->  uint8 RGB
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

PAGE = (12, 14, 19)
CREAM = (240, 233, 214)
CREAM_SHADE = (214, 205, 180)
VEIN = (84, 78, 60)
OUTLINE = (96, 90, 72)
LEAF = (66, 108, 58)
LEAF_LIGHT = (122, 166, 96)
LEAF_OUT = (160, 192, 128)
STEM = (96, 138, 74)
ORANGE = (232, 150, 62)
DOT = (40, 43, 56)
RING = (92, 96, 112)

SS = 2  # supersampling


def rot(pts, ang, ox=0.0, oy=0.0):
    c, s = np.cos(ang), np.sin(ang)
    p = np.asarray(pts, float)
    return np.stack([p[:, 0] * c - p[:, 1] * s + ox, p[:, 0] * s + p[:, 1] * c + oy], 1)


def wobble(pts, rng, amt=1.4):
    return np.asarray(pts) + rng.normal(0, amt, np.shape(pts))


class Pen:
    def __init__(self, w, h, seed):
        self.rng = np.random.default_rng(seed)
        self.im = Image.new("RGB", (w * SS, h * SS), PAGE)
        self.d = ImageDraw.Draw(self.im)

    def poly(self, pts, fill=None, outline=None, width=2):
        p = [tuple(map(float, q * SS)) for q in pts]
        if fill is not None:
            self.d.polygon(p, fill=fill)
        if outline is not None:
            self.d.line(p + [p[0]], fill=outline, width=int(width * SS), joint="curve")

    def line(self, pts, fill, width=2):
        p = [tuple(map(float, q * SS)) for q in np.asarray(pts)]
        self.d.line(p, fill=fill, width=int(width * SS), joint="curve")

    def dot(self, x, y, r, fill):
        self.d.ellipse([(x - r) * SS, (y - r) * SS, (x + r) * SS, (y + r) * SS], fill=fill)

    # ---- shapes ----
    def petal(self, cx, cy, ang, L, W, shade=0.0, round_tip=0.85):
        t = np.linspace(0, 1, 18)
        half = W * np.sin(np.pi * t ** 0.75) ** round_tip * (1 - 0.15 * t)
        pts = np.vstack([np.stack([t * L, half], 1), np.stack([t[::-1] * L, -half[::-1]], 1)])
        pts = wobble(rot(pts, ang, cx, cy), self.rng, 0.9)
        fill = tuple(int(CREAM[i] * (1 - shade) + CREAM_SHADE[i] * shade) for i in range(3))
        self.poly(pts, fill=fill, outline=OUTLINE, width=2.6)
        for k in (-0.35, 0, 0.35):          # veins
            v = np.array([[L * 0.12, k * W * 0.25], [L * 0.7, k * W * 0.7]])
            self.line(wobble(rot(v, ang, cx, cy), self.rng, 0.5), VEIN, 1)

    def flower(self, cx, cy, size, n=5, rot0=0.0, layers=1, stamens=14):
        for layer in range(layers):
            off = rot0 + layer * np.pi / n
            L = size * (1 - 0.18 * layer)
            for i in range(n):
                a = off + i * 2 * np.pi / n + self.rng.normal(0, 0.06)
                self.petal(cx, cy, a, L * self.rng.uniform(0.92, 1.05), L * 0.42,
                           shade=0.25 + 0.3 * layer)
        self.dot(cx, cy, size * 0.12, (226, 200, 120))
        for _ in range(stamens):
            a = self.rng.uniform(0, 2 * np.pi)
            r = size * self.rng.uniform(0.15, 0.32)
            x, y = cx + r * np.cos(a), cy + r * np.sin(a)
            self.line([[cx, cy], [x, y]], (200, 170, 100), 1)
            self.dot(x, y, 2.2, ORANGE)

    def leaf(self, bx, by, ang, L, W):
        t = np.linspace(0, 1, 20)
        half = W * np.sin(np.pi * t) ** 0.9 * (1 - 0.3 * t)
        bend = 0.12 * L * np.sin(np.pi * t)
        pts = np.vstack([np.stack([t * L, half + bend], 1), np.stack([t[::-1] * L, -half[::-1] + bend[::-1]], 1)])
        pts = wobble(rot(pts, ang, bx, by), self.rng, 0.8)
        self.poly(pts, fill=LEAF, outline=LEAF_OUT, width=1.5)
        vein = np.stack([t * L * 0.95, bend * 0.9], 1)
        self.line(rot(vein, ang, bx, by), LEAF_LIGHT, 1.4)
        for k in np.linspace(0.2, 0.75, 4):      # side veins
            for sgn in (-1, 1):
                p0 = np.array([[k * L, bend[int(k * 19)]], [(k + 0.14) * L, sgn * W * 0.55 * np.sin(np.pi * (k + 0.14))]])
                self.line(rot(p0, ang, bx, by), (92, 138, 78), 1)

    def stem(self, pts, leaves=True, width=3.5, leaf_len=70):
        pts = np.asarray(pts, float)
        self.line(pts, STEM, width)
        if leaves:
            for i in range(2, len(pts) - 1, 2):
                a = np.arctan2(*(pts[i + 1] - pts[i - 1])[::-1])
                side = 1 if (i // 2) % 2 else -1
                self.leaf(pts[i][0], pts[i][1], a + side * self.rng.uniform(0.7, 1.0),
                          leaf_len * self.rng.uniform(0.8, 1.1), leaf_len * 0.24)


def bezier(p0, p1, p2, n=12):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * np.array(p0) + 2 * (1 - t) * t * np.array(p1) + t ** 2 * np.array(p2)


def star(pen, x, y, r, color):
    pen.line([[x - r, y], [x + r, y]], color, 1.4)
    pen.line([[x, y - r], [x, y + r]], color, 1.4)


def build_page(w=810, h=780, seed=4):
    pen = Pen(w, h, seed)
    rng = pen.rng

    # dot-grid notebook paper
    for gy in range(24, h, 36):
        for gx in range(24, w, 36):
            pen.dot(gx, gy, 2.0, DOT)

    # spiral binding on the left edge
    for y in range(40, h, 62):
        pen.dot(16, y, 9, (8, 9, 13))
        pen.line([[2, y - 5], [26, y + 3]], RING, 3)

    # big magnolia, top-left (leaves first so petals sit on top)
    for a, L in [(-0.5, 120), (0.3, 130), (1.2, 115), (2.2, 110), (3.0, 100)]:
        pen.leaf(120, 160, a + np.pi * 0.0, L, 28)
    pen.flower(120, 160, 88, n=6, rot0=0.2, layers=2, stamens=18)

    # spray on the right edge: a stem with three small flowers
    s = bezier((795, -10), (690, 220), (730, 520), 16)
    pen.stem(s, leaves=True, leaf_len=72)
    for (px, py, sz, r0) in [(742, 70, 42, 0.4), (700, 250, 46, 1.1), (722, 440, 40, 2.0)]:
        pen.flower(px, py, sz, n=5, rot0=r0, layers=1, stamens=10)

    # bottom-left cluster
    s2 = bezier((40, 790), (110, 640), (250, 600), 12)
    pen.stem(s2, leaves=True, leaf_len=64)
    pen.flower(70, 690, 46, n=5, rot0=0.7, layers=1, stamens=10)
    pen.flower(210, 740, 34, n=5, rot0=1.5, layers=1, stamens=8)

    # bottom-right bud branch
    s3 = bezier((810, 790), (700, 700), (640, 610), 12)
    pen.stem(s3, leaves=True, leaf_len=60)
    pen.flower(745, 735, 38, n=5, rot0=0.2, layers=1, stamens=9)

    # doodles: stars, dots, tiny leaves
    for _ in range(26):
        x, y = rng.uniform(40, w), rng.uniform(0, h)
        k = rng.integers(0, 3)
        if k == 0:
            star(pen, x, y, rng.uniform(4, 8), (170, 165, 150))
        elif k == 1:
            pen.dot(x, y, rng.uniform(2, 4), (190, 180, 150))
        else:
            pen.leaf(x, y, rng.uniform(0, 6.28), 22, 6)

    im = pen.im.filter(ImageFilter.GaussianBlur(0.5)).resize((w, h), Image.LANCZOS)
    return np.asarray(im)


if __name__ == "__main__":
    Image.fromarray(build_page()).save("page.png")
