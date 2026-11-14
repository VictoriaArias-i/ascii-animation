"""Static cut-out of the girl: removes purple sticker, outline and grid paper.

alpha = make_alpha(rgb)  ->  float32 0..1, same size as rgb
"""
import numpy as np
import cv2

ROI = (150, 110, 1010, 940)  # x0, y0, x1, y1 in rot.png coordinates


def make_alpha(rgb):
    H, W = rgb.shape[:2]
    x0, y0, x1, y1 = ROI
    sub = rgb[y0:y1, x0:x1].astype(np.int32)
    r, g, b = sub[..., 0], sub[..., 1], sub[..., 2]
    mx, mn = sub.max(-1), sub.min(-1)
    lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
    sat = (mx - mn) / np.maximum(mx, 1)

    purple = (b > r + 10) & (b > g + 14)
    paper = (lum > 0.55) & (sat < 0.16) & ~purple

    cand = (purple | paper).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    big = np.zeros(n, bool)
    h, w = cand.shape
    for i in range(1, n):
        x, y, cw, chh, area = stats[i]
        touches = x == 0 or y == 0 or x + cw >= w or y + chh >= h
        big[i] = area > 25000 or (touches and area > 3000)
    bigmask = big[lab]
    bg = purple | (paper & bigmask)

    char = (~bg).astype(np.uint8)
    char = cv2.morphologyEx(char, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(char, connectivity=8)
    keep = np.zeros(n, bool)
    biggest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    keep[biggest] = True
    for i in range(1, n):          # small pieces right next to the main body (hair curls)
        if stats[i, cv2.CC_STAT_AREA] > 40:
            comp = (lab == i).astype(np.uint8)
            near = cv2.dilate((lab == biggest).astype(np.uint8), np.ones((9, 9), np.uint8))
            if (comp & near).any():
                keep[i] = True
    char = keep[lab].astype(np.uint8)
    char = cv2.erode(char, np.ones((3, 3), np.uint8))          # drop lavender fringe
    a = np.zeros((H, W), np.float32)
    a[y0:y1, x0:x1] = cv2.GaussianBlur(char.astype(np.float32), (0, 0), 0.9)
    return a


if __name__ == "__main__":
    rgb = cv2.cvtColor(cv2.imread("rot.png"), cv2.COLOR_BGR2RGB)
    a = make_alpha(rgb)
    np.save("alpha.npy", a)
    x0, y0, x1, y1 = (190, 130, 1000, 910)
    over = rgb[y0:y1, x0:x1].astype(np.float32)
    al = a[y0:y1, x0:x1, None]
    checker = np.indices(over.shape[:2]).sum(0) // 20 % 2
    bgc = np.where(checker[..., None] == 0, [30, 30, 40], [60, 60, 75]).astype(np.float32)
    out = over * al + bgc * (1 - al)
    cv2.imwrite("alpha_check.png", cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_RGB2BGR))
