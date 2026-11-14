"""Turn the animated drawing into colour ASCII art (GIF + MP4).

Run: python ascii_girl.py                -> girl_ascii.gif / girl_ascii.mp4
     python ascii_girl.py --terminal     -> play in your terminal
Needs animate.py, cutout.py, sketchbook.py and rot.png next to it, plus
numpy, opencv-python, pillow and ffmpeg.
"""
import argparse
import subprocess
import sys
import time
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from animate import Animator

COLS = 260
RAMP = " .:;ox%#@"
FONTS = [  # first one that exists wins (Linux, Windows, macOS)
    "DejaVuSansMono.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "C:/Windows/Fonts/consola.ttf", "C:/Windows/Fonts/cour.ttf",
    "/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/Courier.ttc",
]
FONT_SIZE = 7
BG = np.array([9, 11, 16], dtype=np.float32)
FPS = 20
FRAMES = 80                       # 4 seconds, loops seamlessly
LO, HI = 0.06, 0.80               # fixed brightness range (no per-frame stretch -> no flicker)


def to_cells(frame):
    h, w = frame.shape[:2]
    rows = int(round(h / w * COLS / 2))
    small = cv2.resize(frame, (COLS, rows), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    lum = small @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    dens = np.clip((lum - LO) / (HI - LO), 0, 1) ** 0.85
    green = (small[..., 1] > small[..., 0] + 0.03) & (small[..., 1] > small[..., 2] + 0.05) & (lum < 0.45) & (lum > 0.07)
    dens = np.where(green, np.maximum(dens, 0.40), dens)       # keep dark green hair and leaves visible
    idx = (dens * (len(RAMP) - 1)).round().astype(int).clip(0, len(RAMP) - 1)
    peak = np.maximum(small.max(-1, keepdims=True), 1e-3)
    col = small * np.maximum(1.0, 0.88 / peak)                  # lift dark colours so they glow
    return idx, np.clip(col, 0, 1)


def make_atlas():
    font = None
    for path in FONTS:
        try:
            font = ImageFont.truetype(path, FONT_SIZE)
            break
        except OSError:
            pass
    if font is None:
        raise SystemExit("No monospace font found: add its path to FONTS in ascii_girl.py")
    cw = int(round(font.getlength("M")))
    ch = FONT_SIZE + 2
    atlas = np.zeros((len(RAMP), ch, cw), dtype=np.float32)
    for i, c in enumerate(RAMP):
        im = Image.new("L", (cw, ch), 0)
        ImageDraw.Draw(im).text((0, 0), c, font=font, fill=255)
        atlas[i] = np.asarray(im, dtype=np.float32) / 255
    return atlas, cw, ch


def render(frame, atlas, cw, ch):
    idx, col = to_cells(frame)
    rows, cols = idx.shape
    g = atlas[idx][..., None]
    img = g * (col[:, :, None, None, :] * 255) + (1 - g) * BG
    img = img.transpose(0, 2, 1, 3, 4).reshape(rows * ch, cols * cw, 3)
    return img.astype(np.uint8)


def play_terminal(anim):
    sys.stdout.write("\x1b[2J\x1b[?25l")
    try:
        i = 0
        while True:
            idx, col = to_cells(anim.frame((i % FRAMES) / FRAMES))
            out = ["\x1b[H"]
            for r in range(idx.shape[0]):
                row = []
                for c in range(idx.shape[1]):
                    R, G, B = (col[r, c] * 255).astype(int)
                    row.append(f"\x1b[38;2;{R};{G};{B}m{RAMP[idx[r, c]]}")
                out.append("".join(row) + "\x1b[0m\n")
            sys.stdout.write("".join(out))
            sys.stdout.flush()
            time.sleep(1 / FPS)
            i += 1
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\x1b[0m\x1b[?25h\n")


def run(cmd):
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--terminal", action="store_true")
    ap.add_argument("--out", default="girl_ascii")
    ap.add_argument("--gif-width", type=int, default=1040)
    args = ap.parse_args()
    anim = Animator()
    if args.terminal:
        return play_terminal(anim)

    atlas, cw, ch = make_atlas()
    frames = [render(anim.frame(i / FRAMES), atlas, cw, ch) for i in range(FRAMES)]
    h, w = frames[0].shape[:2]
    h, w = h - h % 2, w - w % 2

    # MP4 (full size, 5 loops)
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
                          "-vf", f"loop=loop=4:size={FRAMES}",
                          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "17", args.out + ".mp4"],
                         stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f[:h, :w]).tobytes())
    p.stdin.close()
    p.wait()

    # GIF (one loop, shared palette, scaled down a bit to keep the file small)
    vf = f"trim=end_frame={FRAMES},scale={args.gif_width}:-1:flags=lanczos"
    run(["ffmpeg", "-v", "error", "-y", "-i", args.out + ".mp4", "-vf", vf + ",palettegen=max_colors=128:stats_mode=full", "pal.png"])
    run(["ffmpeg", "-v", "error", "-y", "-i", args.out + ".mp4", "-i", "pal.png", "-lavfi",
         vf + "[x];[x][1:v]paletteuse=dither=none", "-loop", "0", args.out + ".gif"])
    print("saved", args.out + ".gif", args.out + ".mp4")


if __name__ == "__main__":
    main()
