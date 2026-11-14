"""Animate the still drawing: slow waving hand, blinking, swaying hair. Face stays still.

frame(t) -> uint8 RGB of the bust over a sketchbook page; t in [0, 1) loops seamlessly.
"""
import numpy as np
import cv2

from cutout import make_alpha
from sketchbook import build_page

SRC = "rot.png"
CROP = (190, 130, 1000, 910)           # x0, y0, x1, y1
PIVOT = np.array([415.0, 765.0])       # wrist
HAND_DIR = np.array([-0.70, -0.70])    # wrist -> fingers
HAND_SEED = (330, 680)                 # a pixel inside the hand
HAND_CYCLES = 2                        # waves per loop (loop = 4 s -> 0.5 Hz, slow)
FACE = dict(cx=625, cy=468, rx=170, ry=140)   # hair never moves inside this oval
EYES = [  # centre x, y, half width, half height, skin sample point
    dict(cx=533, cy=414, a=38, b=22, skin=(540, 452)),
    dict(cx=681, cy=411, a=44, b=19, skin=(690, 446)),
]
LASH = np.array([40, 30, 20], dtype=np.float32)


def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


class Animator:
    def __init__(self):
        self.img = cv2.cvtColor(cv2.imread(SRC), cv2.COLOR_BGR2RGB)
        self.H, self.W = self.img.shape[:2]
        self.alpha = make_alpha(self.img)
        self.base4 = np.dstack([self.img.astype(np.float32), self.alpha * 255])
        self.ys, self.xs = np.mgrid[0:self.H, 0:self.W].astype(np.float32)
        x0, y0, x1, y1 = CROP
        self.page = build_page(x1 - x0, y1 - y0).astype(np.float32)
        self._prep_hand()
        self._prep_hair()
        self._prep_eyes()

    # ---------- hand ----------
    def _prep_hand(self):
        r, g, b = [self.img[..., i].astype(np.int32) for i in range(3)]
        purple = (b > r + 12) & (b > g + 12)
        paper = (r > 215) & (g > 215)
        proj = (self.xs - PIVOT[0]) * HAND_DIR[0] + (self.ys - PIVOT[1]) * HAND_DIR[1]
        cand = (~purple) & (~paper) & (proj > 12) & (self.xs < 446) & (self.ys > 585) & (self.ys < 800)
        cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, lab = cv2.connectedComponents(cand)
        hand = (lab == lab[HAND_SEED[1], HAND_SEED[0]]).astype(np.uint8)
        hand = cv2.dilate(hand, np.ones((5, 5), np.uint8))
        hand = cv2.morphologyEx(hand, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        self.hand_mask = hand
        grow = cv2.dilate(hand, np.ones((15, 15), np.uint8))
        self.grow = grow
        rgb_bg = cv2.inpaint(self.img, grow, 9, cv2.INPAINT_TELEA).astype(np.float32)
        a_bg = self.alpha * (1 - grow) * 255          # where the hand left, the page shows through
        self.bg4 = np.dstack([rgb_bg, a_bg])
        keep = 1.0 - grow.astype(np.float32)
        self.keep = cv2.GaussianBlur(keep, (0, 0), 2.0)[..., None]

    def hand_layer(self, ang):
        M = cv2.getRotationMatrix2D((float(PIVOT[0]), float(PIVOT[1])), ang, 1.0)
        rot4 = cv2.warpAffine(self.base4, M, (self.W, self.H), flags=cv2.INTER_CUBIC)
        m = cv2.warpAffine(self.hand_mask.astype(np.float32), M, (self.W, self.H), flags=cv2.INTER_LINEAR)
        m = cv2.GaussianBlur(m, (0, 0), 1.2)[..., None]
        return rot4, m

    # ---------- hair ----------
    def _prep_hair(self):
        hsv = cv2.cvtColor(self.img, cv2.COLOR_RGB2HSV).astype(np.float32)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        green = (h > 30) & (h < 85) & (s > 70) & (v < 170)
        green &= (self.ys < 760)
        green[(self.xs < 440) & (self.ys > 570)] = False   # keep scrunchie/hand out
        m = cv2.morphologyEx(green.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        hair = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 3.0)
        face = (((self.xs - FACE["cx"]) / FACE["rx"]) ** 2 + ((self.ys - FACE["cy"]) / FACE["ry"]) ** 2 < 1)
        face = cv2.GaussianBlur(face.astype(np.float32), (0, 0), 14.0)
        self.hair = hair * (1 - np.clip(face * 1.6, 0, 1))
        self.hair_amp = 2.0 + 11.0 * smoothstep((self.ys - 330) / 360.0)

    def warp_hair(self, img4, t):
        ph = 2 * np.pi * t
        side = np.where(self.xs < 620, 0.0, 1.9)
        sway = np.sin(ph - self.ys / 140.0 + side) + 0.35 * np.sin(2 * ph - self.ys / 90.0 + side * 1.7)
        dx = self.hair * self.hair_amp * sway
        dy = self.hair * self.hair_amp * 0.25 * np.sin(ph + self.xs / 120.0)
        return cv2.remap(img4, (self.xs - dx).astype(np.float32), (self.ys - dy).astype(np.float32),
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # ---------- eyes ----------
    def _prep_eyes(self):
        for e in EYES:
            x, y = e["skin"]
            patch = self.img[y - 5:y + 6, x - 12:x + 13].reshape(-1, 3)
            e["color"] = np.median(patch, axis=0).astype(np.float32)

    def blink(self, rgb, c):
        if c <= 0.02:
            return rgb
        out = rgb.copy()
        for e in EYES:
            u = (self.xs - e["cx"]) / e["a"]
            vv = (self.ys - e["cy"]) / e["b"]
            inside = np.clip((1.0 - (u ** 2 + vv ** 2)) * 6, 0, 1)
            ylid = e["cy"] - e["b"] + c * 2 * e["b"] + 0.30 * e["b"] * c * (1 - np.clip(u, -1, 1) ** 2)
            cover = np.clip((ylid - self.ys) / 2.0 + 0.5, 0, 1) * inside
            out = out * (1 - cover[..., None]) + e["color"] * cover[..., None]
            line = np.clip(1.0 - np.abs(self.ys - ylid) / 3.0, 0, 1) * inside * min(1.0, c * 3)
            out = out * (1 - line[..., None]) + LASH * line[..., None]
        return out

    @staticmethod
    def blink_amount(t):
        """0 open ... 1 closed; one blink per loop."""
        x = (t - 0.62) / 0.10
        return float(np.clip(1 - abs(x), 0, 1)) if -1 < x < 1 else 0.0

    # ---------- frame ----------
    def frame(self, t):
        ang = -3.0 + 13.0 * np.sin(2 * np.pi * HAND_CYCLES * t)
        rot4, hm = self.hand_layer(ang)
        img4 = self.bg4 * (1 - hm) + rot4 * hm
        img4 = img4 * (1 - self.keep) + self.base4 * self.keep
        img4 = self.warp_hair(img4.astype(np.float32), t)
        rgb = self.blink(img4[..., :3], self.blink_amount(t))
        x0, y0, x1, y1 = CROP
        rgb = rgb[y0:y1, x0:x1]
        a = np.clip(img4[y0:y1, x0:x1, 3:4] / 255.0, 0, 1)
        out = rgb * a + self.page * (1 - a)
        return np.clip(out, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    from PIL import Image
    a = Animator()
    strip = [a.frame(t) for t in (0.0, 0.125, 0.25, 0.62)]
    Image.fromarray(np.hstack([s[::2, ::2] for s in strip])).save("preview.png")
