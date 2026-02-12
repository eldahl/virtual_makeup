import numpy as np
import mediapipe as mp
import cv2
import os
import urllib.request

# MediaPipe Tasks API imports
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# landmarks of features from mediapipe
face_points = {
    "BLUSH_LEFT": [50],
    "BLUSH_RIGHT": [280],
    "LEFT_EYE": [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7, 33],
    "RIGHT_EYE": [362, 298, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382, 362],
    "EYELINER_LEFT": [243, 112, 26, 22, 23, 24, 110, 25, 226, 130, 33, 7, 163, 144, 145, 153, 154, 155, 133, 243],
    "EYELINER_RIGHT": [463, 362, 382, 381, 380, 374, 373, 390, 249, 263, 359, 446, 255, 339, 254, 253, 252, 256, 341, 463],
    "EYESHADOW_LEFT": [226, 247, 30, 29, 27, 28, 56, 190, 243, 173, 157, 158, 159, 160, 161, 246, 33, 130, 226],
    "EYESHADOW_RIGHT": [463, 414, 286, 258, 257, 259, 260, 467, 446, 359, 263, 466, 388, 387, 386, 385, 384, 398, 362, 463],
    "FACE": [152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 389, 454, 323, 401, 361, 435, 288, 397, 365, 379, 378, 400, 377, 152],
    "LIP_UPPER": [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 308, 415, 310, 312, 13, 82, 81, 80, 191, 78],
    "LIP_LOWER": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 402, 317, 14, 87, 178, 88, 95, 78, 61],
    "EYEBROW_LEFT": [55, 107, 66, 105, 63, 70, 46, 53, 52, 65, 55],
    "EYEBROW_RIGHT": [285, 336, 296, 334, 293, 300, 276, 283, 295, 285]
}

# Model file path
MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"


def ensure_model_exists():
    """Download the face landmarker model if it doesn't exist."""
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading face landmarker model to {MODEL_PATH}...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded successfully.")


def _normalized_to_pixel_coordinates(normalized_x, normalized_y, image_width, image_height):
    """Convert normalized coordinates to pixel coordinates."""
    x_px = min(int(normalized_x * image_width), image_width - 1)
    y_px = min(int(normalized_y * image_height), image_height - 1)
    if 0 <= x_px < image_width and 0 <= y_px < image_height:
        return (x_px, y_px)
    return None


# Global face landmarker instance for reuse
_face_landmarker = None
_frame_timestamp = 0


def get_face_landmarker():
    """Get or create a FaceLandmarker instance using VIDEO mode for better tracking."""
    global _face_landmarker
    if _face_landmarker is None:
        ensure_model_exists()
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,  # VIDEO mode enables tracking
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False
        )
        _face_landmarker = vision.FaceLandmarker.create_from_options(options)
    return _face_landmarker


def close_face_landmarker():
    """Close the global face landmarker instance."""
    global _face_landmarker, _frame_timestamp
    if _face_landmarker is not None:
        _face_landmarker.close()
        _face_landmarker = None
    _frame_timestamp = 0


# to display image in cv2 window
def show_image(image: np.array, msg: str = "Loaded Image"):
    """
    image : image as np array
    msg : cv2 window name
    """
    image_copy = image.copy()
    cv2.imshow(msg, image_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def read_landmarks(image: np.array):
    """
    Read facial landmarks from an image using MediaPipe Tasks API.
    Uses VIDEO mode for efficient tracking between frames.
    
    image : image as np.array (BGR format from OpenCV)
    Returns: dict mapping landmark index to (x, y) pixel coordinates
    """
    global _frame_timestamp
    landmark_coordinates = {}
    
    # Convert BGR to RGB for MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    
    # Get face landmarker and detect using VIDEO mode
    landmarker = get_face_landmarker()
    _frame_timestamp += 33  # ~30fps increment
    result = landmarker.detect_for_video(mp_image, _frame_timestamp)
    
    # Check if any faces were detected
    if not result.face_landmarks:
        return landmark_coordinates
    
    # Get the first face's landmarks
    face_landmarks = result.face_landmarks[0]
    
    # Convert normalized points to pixel coordinates
    height, width = image.shape[:2]
    for idx, landmark in enumerate(face_landmarks):
        landmark_px = _normalized_to_pixel_coordinates(
            landmark.x, landmark.y, width, height
        )
        # Create a map of facial landmarks to (x,y) coordinates
        if landmark_px:
            landmark_coordinates[idx] = landmark_px
    
    return landmark_coordinates


# based on input facial features create make w.r.to colors
def add_mask(
    mask: np.array, idx_to_coordinates: dict, face_connections: list, colors: list
):
    """
    mask: image filled with 0's
    idx_to_coordinates : dict with (x,y) cordinates for each face landmarks
    face_connections : list of (x,y) cordinates for each facial features
    colors : list of [B,G,R] color for each features
    """
    for i, connection in enumerate(face_connections):
        # extract (x,y) w.r.to image for each cordinates
        try:
            points = np.array([idx_to_coordinates[idx] for idx in connection])
            # make a shape of feature in the mask and add color
            cv2.fillPoly(mask, [points], colors[i])
        except KeyError:
            # Skip if landmarks not found
            continue

    # smoothening of image - use smaller kernel for speed
    mask = cv2.GaussianBlur(mask, (5, 5), 3)
    return mask
