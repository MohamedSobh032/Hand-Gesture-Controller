import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

########################################### SEGMENTATION BASED ON COLOR ##########################################

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


############################################# SEGMENTATION BASED OTSU ############################################

# img_gaussian = cle.gaussian_blur(img, sigma_x=2, sigma_y=2, sigma_z=2)
# img_maxima_locations = cle.detect_maxima_box(img_gaussian, radius_x=0, radius_y=0, radius_z=0)
# number_of_maxima_locations = cle.sum_of_all_pixels(img_maxima_locations)
# segment the image using OTSU
# roi_otsu = util.thresholding_otsu(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)).astype(np.uint8)
# cv2.imshow('ROI of OTSU', roi_otsu)

############################################# K-Means Based on GRAD ##############################################
def segment_hand_with_gradients(frame):

    # Fixed ROI coordinates
    roi_x1, roi_y1, roi_x2, roi_y2 = 50, 50, 250, 250

    # Extract ROI from the frame
    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

    # Convert ROI to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Compute gradient magnitudes using Sobel filters
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    gradient_magnitude = cv2.magnitude(grad_x, grad_y)

    # Normalize the gradient magnitude
    gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # Combine grayscale intensity and gradient magnitude as features
    pixel_features = np.stack((gray.flatten(), gradient_magnitude.flatten()), axis=1).astype(np.float32)

    # K-means clustering
    k = 2  # Two clusters: hand and background
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_features, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Reshape labels to match the ROI dimensions
    labels = labels.reshape(gray.shape)

    # Determine which cluster is likely the hand
    cluster_hand = 1 if np.sum(labels == 1) > np.sum(labels == 0) else 0

    # Create a binary mask for the hand cluster
    mask = (labels == cluster_hand).astype(np.uint8) * 255

    # Map the mask back to the full frame size
    full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    full_mask[roi_y1:roi_y2, roi_x1:roi_x2] = mask

    # Apply the mask to the original frame
    segmented_hand = cv2.bitwise_and(frame, frame, mask=full_mask)

    return segmented_hand, full_mask


############################################## K-Means Based on X-Y ##############################################
def segment_hand(frame, n_clusters=2):
    """
    Segments the hand from a video frame using clustering.

    Args:
        frame (np.array): Input video frame (BGR format).
        n_clusters (int): Number of clusters for K-Means (default: 2).

    Returns:
        segmented_mask (np.array): Binary mask of the segmented hand.
        segmented_image (np.array): Colorized segmented result.
    """
    # Step 1: Convert to HSV color space and blur
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blurred_frame = cv2.GaussianBlur(hsv_frame, (11, 11), 0)

    # Step 2: Prepare feature space (HSV + X, Y coordinates)
    height, width, _ = frame.shape
    X = []
    for y in range(height):
        for x in range(width):
            h, s, v = blurred_frame[y, x]
            X.append([h, s, v, y, x])  # Combine HSV and spatial coordinates

    X = np.array(X)

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 3: Apply K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_scaled)
    labels = kmeans.labels_.reshape(height, width)

    # Step 4: Identify the cluster corresponding to the hand
    # Assume the largest cluster near the center of the frame is the hand
    center_y, center_x = height // 2, width // 2
    center_label = labels[center_y, center_x]

    # Generate a binary mask for the hand
    segmented_mask = (labels == center_label).astype(np.uint8) * 255

    # Step 5: Post-process the mask (Morphological operations)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    segmented_mask = cv2.morphologyEx(segmented_mask, cv2.MORPH_CLOSE, kernel)
    segmented_mask = cv2.morphologyEx(segmented_mask, cv2.MORPH_OPEN, kernel)

    # Step 6: Generate segmented image
    segmented_image = cv2.bitwise_and(frame, frame, mask=segmented_mask)
    
    return segmented_mask, segmented_image