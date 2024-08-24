
import cv2
import numpy as np

def calculate_distance(pt1, pt2):
    return np.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)

def crop_and_transform(image, points):
    # Convert points to numpy array
    pts = np.array(points, dtype="float32")

    # Calculate width of the new image (average of top and bottom width)
    widthA = calculate_distance(pts[0], pts[1])
    widthB = calculate_distance(pts[2], pts[3])
    maxWidth = max(int(widthA), int(widthB))

    # Calculate height of the new image (average of left and right height)
    heightA = calculate_distance(pts[0], pts[3])
    heightB = calculate_distance(pts[1], pts[2])
    maxHeight = max(int(heightA), int(heightB))

    # Define the destination points which correspond to the corners of the output rectangle
    dst_pts = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    # Calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(pts, dst_pts)

    # Apply the perspective transformation
    transformed = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return transformed