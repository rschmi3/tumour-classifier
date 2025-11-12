import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern


def extract_hog_features(images):
    """Extract HOG features from images"""
    print("Extracting HOG features...")
    features = []
    for img in images:
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Extract HOG features
        fd = hog(
            gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=False,
        )
        features.append(fd)
    return np.array(features)


def extract_lbp_features(images):
    """Extract Local Binary Pattern features"""
    print("Extracting LBP features...")
    features = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # LBP parameters
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
        # Create histogram
        hist, _ = np.histogram(
            lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2)
        )
        # Normalize
        hist = hist.astype("float")
        hist /= hist.sum() + 1e-6
        features.append(hist)
    return np.array(features)


def extract_color_features(images):
    """Extract color histogram features"""
    print("Extracting color histogram features...")
    features = []
    for img in images:
        # Calculate histogram for each channel
        hist_features = []
        for i in range(3):  # RGB channels
            hist = cv2.calcHist([img], [i], None, [32], [0, 256])
            hist = hist.flatten()
            hist = hist / (hist.sum() + 1e-6)  # Normalize
            hist_features.extend(hist)
        features.append(hist_features)
    return np.array(features)


def extract_combined_features(images):
    """Combine multiple feature extraction methods"""
    print("Extracting combined features (HOG + LBP + Color)...")
    hog_features = extract_hog_features(images)
    lbp_features = extract_lbp_features(images)
    color_features = extract_color_features(images)

    # Concatenate all features
    combined = np.hstack([hog_features, lbp_features, color_features])
    print(f"Combined feature shape: {combined.shape}")
    return combined
