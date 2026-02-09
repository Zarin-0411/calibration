import cv2
import numpy as np
from datetime import datetime

# ============= CONFIG =============

IMAGE_PATH = "calibration/calib_image.jpeg"  
SAVE_ON_ESC = True
TEXT_COLOR = (255, 0, 0)        # Blue (BGR)
DOT_COLOR  = (0, 0, 255)        # Red dot (BGR)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 2

# Paste your Homography from Step 5 here:
H = np.array([
    [1.46942136e+00, -4.01685843e-01, -1.14808542e+03],
    [-1.92001571e-02, -2.15826917e-01,  1.24984536e+02],
    [ 6.25869811e-06, -7.04627411e-04,  1.00000000e+00]
], dtype=np.float64)

# Optional: predefined pixel points to annotate in batch
PREDEFINED_POINTS = []  # e.g. [(867, 237), (1177, 438), (1497, 538), (888, 741)]

# ==================================

def pixel_to_robot(u, v, H):
    p = np.array([u, v, 1.0], dtype=np.float64)
    pr = H @ p
    pr = pr / pr[2]
    return float(pr[0]), float(pr[1])  # X, Y in mm

def format_pair(u, v, X, Y):
    # Matches your style: "867,237 -> 350,75" with integer-like formatting
    def fmt_num(z):  # keep integers if close; otherwise one decimal
        if abs(z - round(z)) < 1e-3:
            return f"{int(round(z))}"
        return f"{z:.1f}"
    return f"{fmt_num(u)},{fmt_num(v)} -> {fmt_num(X)},{fmt_num(Y)}"

def put_label(img, x, y, text, color=TEXT_COLOR):
    # offset so text doesn’t overlap the dot
    offset_x, offset_y = 14, -14
    org = (int(x + offset_x), int(y + offset_y))
    cv2.putText(img, text, org, FONT, FONT_SCALE, color, THICKNESS, cv2.LINE_AA)

def draw_click(img, x, y, text):
    # mark the point and write the label
    cv2.circle(img, (int(x), int(y)), 6, DOT_COLOR, -1, lineType=cv2.LINE_AA)
    put_label(img, x, y, text)

# ------------- MAIN --------------

img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Could not read {IMAGE_PATH}. Put it next to the script.")

canvas = img.copy()

# Batch mode (optional)
for (u, v) in PREDEFINED_POINTS:
    X, Y = pixel_to_robot(u, v, H)
    text = format_pair(u, v, X, Y)
    draw_click(canvas, u, v, text)
    print(text)

# Interactive mode
def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        X, Y = pixel_to_robot(x, y, H)
        text = format_pair(x, y, X, Y)
        draw_click(canvas, x, y, text)
        print(text)

win = "Click to annotate (ESC saves & exits)"
cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback(win, on_mouse)

while True:
    cv2.imshow(win, canvas)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        if SAVE_ON_ESC:
            out_name = f"annotated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(out_name, canvas)
            print(f"Saved: {out_name}")
        break

cv2.destroyAllWindows()