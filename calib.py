import cv2

points_img = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at pixel: ({x}, {y})")
        points_img.append((x, y))

# Load your image
img = cv2.imread('calibration/calib_image.jpeg')
img_show = img.copy()

cv2.namedWindow('calib')
cv2.setMouseCallback('calib', click_event)

while True:
    cv2.imshow('calib', img_show)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC to quit
        break

cv2.destroyAllWindows()
print("Collected points:", points_img)