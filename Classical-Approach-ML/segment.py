import cv2
import numpy as np

def color_seg(frame):

    # Convert the frame to RGB color space
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    r = rgb_frame[:, :, 0]
    g = rgb_frame[:, :, 1]
    b = rgb_frame[:, :, 2]
    r_norm = r / (r + g + b + 1e-6)
    g_norm = g / (r + g + b + 1e-6)
    rgb_mask = (r_norm / (g_norm + 1e-6)) > 1.185
    rgb_mask = rgb_mask.astype(np.uint8) * 255

    # Convert the frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h = hsv_frame[:, :, 0]
    s = hsv_frame[:, :, 1]
    h_mask = ((h > 0) & (h < 14)) | ((h > 167) & (h < 180))
    s_mask = (s > 50) & (s < 155)
    hsv_mask = h_mask & s_mask
    hsv_mask = hsv_mask.astype(np.uint8) * 255

    # Convert the frame to YCrCb color space
    ycrcb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    cr = ycrcb_frame[:, :, 1]
    cb = ycrcb_frame[:, :, 2]
    cr_mask = (cr > 133) & (cr < 173)
    cb_mask = (cb > 77) & (cb < 127)
    ycrcb_mask = cr_mask & cb_mask
    ycrcb_mask = ycrcb_mask.astype(np.uint8) * 255

    # Combine the masks
    mask = cv2.bitwise_and(rgb_mask, hsv_mask)
    mask = cv2.bitwise_and(mask, ycrcb_mask)

    return mask

def segment_hands(frame, face_cascade):

    # Mask the frame based on color
    mask = color_seg(frame)

    # Detect faces in the frame
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, 1.1, 3, minSize=(100, 100))
    for (x, y, w, h) in faces:
        center = (x + w // 2, y + h // 2)
        axes = (w // 2, int(h * 0.75))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 0, -1)

    # Apply morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask
