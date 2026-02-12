from utils import (
    face_points, read_landmarks, add_mask, close_face_landmarker,
    cv2, np
)

# features to add makeup
face_elements = [
    "LIP_LOWER",
    "LIP_UPPER",
    "EYEBROW_LEFT",
    "EYEBROW_RIGHT",
    "EYELINER_LEFT",
    "EYELINER_RIGHT",
    "EYESHADOW_LEFT",
    "EYESHADOW_RIGHT",
    "BLUSH_RIGHT",
    "BLUSH_LEFT"
]

# change the color of features
colors_map = {
    # upper lip and lower lips
    "LIP_UPPER": [0, 0, 255],  # Red in BGR
    "LIP_LOWER": [0, 0, 255],  # Red in BGR
    # eyeliner
    "EYELINER_LEFT": [139, 0, 0],  # Dark Blue in BGR
    "EYELINER_RIGHT": [139, 0, 0],  # Dark Blue in BGR
    # eye shadow
    "EYESHADOW_LEFT": [0, 100, 0],  # Dark Green in BGR
    "EYESHADOW_RIGHT": [0, 100, 0],  # Dark Green in BGR
    # eye brow
    "EYEBROW_LEFT": [19, 69, 139],  # Dark Brown in BGR
    "EYEBROW_RIGHT": [19, 69, 139],  # Dark Brown in BGR

    "BLUSH_LEFT": [244, 0, 0],
    "BLUSH_RIGHT": [0, 0, 244],
}


face_connections = [face_points[idx] for idx in face_elements]
colors = [colors_map[idx] for idx in face_elements]

video_capture = cv2.VideoCapture(0)

try:
    while True:
        # read image from camera
        success, image = video_capture.read()
        if not success:
            continue
        image = cv2.flip(image, 1)
        
        # create a empty mask like image
        mask = np.zeros_like(image)
        # extract facial landmarks
        face_landmarks = read_landmarks(image=image)

        # Only apply mask if landmarks were detected
        if face_landmarks:
            # create mask for facial features with color
            mask = add_mask(
                mask,
                idx_to_coordinates=face_landmarks,
                face_connections=face_connections,
                colors=colors
            )
        # combine the image and mask with w.r.to weights
        output = cv2.addWeighted(image, 1.0, mask, 0.2, 1.0)
        cv2.imshow("Feature", output)
        # press q to exit the cv2 window
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    video_capture.release()
    close_face_landmarker()
    cv2.destroyAllWindows()
