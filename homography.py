import numpy as np
import cv2

# Replace ALL values with your own calibration data!
img_pts = np.array([
    [971, 310],
    [708, 247],
    [639, 423],
    [822, 580],
    [1082, 559],
    [1150, 380]
], dtype=np.float32)

robot_pts = np.array([
    [300 , 25],
    [275, -50],
    [325, -75],
    [375, -25],
    [375, 50],
    [325, 75]
    
], dtype=np.float32)

# Compute homography from image to robot plane
H, mask = cv2.findHomography(img_pts, robot_pts, method=0)

print("Homography matrix H:")
print(H)

def pixel_to_robot(u, v, H):
    p = np.array([u, v, 1.0], dtype=np.float32).reshape(3, 1)
    pr = H @ p
    pr /= pr[2, 0]  # normalize
    X = pr[0, 0]
    Y = pr[1, 0]
    return X, Y

X_pred, Y_pred = pixel_to_robot(971, 310, H)
print("Predicted robot coords:", X_pred, Y_pred)
print("Actual robot coords:", 375, 50)
errors = []

for i in range(len(img_pts)):
    u, v = img_pts[i]
    X_real, Y_real = robot_pts[i]
    
    X_pred, Y_pred = pixel_to_robot(u, v, H)
    
    err = np.sqrt((X_pred - X_real)**2 + (Y_pred - Y_real)**2)
    errors.append(err)
    
    print(f"P{i+1} -> Predicted: ({X_pred:.2f}, {Y_pred:.2f}) mm | "
          f"Actual: ({X_real:.2f}, {Y_real:.2f}) mm | "
          f"Error: {err:.2f} mm")

print("\nAverage error:", np.mean(errors), "mm")
print("Maximum error:", np.max(errors), "mm")
