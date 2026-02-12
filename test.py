import cv2
import mediapipe as mp
import numpy as np
import os
import urllib.request
import time

try:
    import pyvirtualcam
    VIRTUALCAM_AVAILABLE = True
except ImportError:
    VIRTUALCAM_AVAILABLE = False
    print("Note: pyvirtualcam not installed. Virtual camera disabled.")
    print("Install with: pip install pyvirtualcam")
    print("Also need v4l2loopback: sudo modprobe v4l2loopback devices=1")

# MediaPipe Tasks API imports
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Model file path and URL
MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

# ============ PERFORMANCE SETTINGS ============
# Set to True for maximum performance on slow machines
PERFORMANCE_MODE = True

# Processing resolution (smaller = faster, but less accurate)
PROCESS_WIDTH = 640 if PERFORMANCE_MODE else 1280
PROCESS_HEIGHT = 480 if PERFORMANCE_MODE else 720


def ensure_model_exists():
    """Download the face landmarker model if it doesn't exist."""
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading face landmarker model to {MODEL_PATH}...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded successfully.")

# ============ MAKEUP SYSTEM ============

# Face landmark regions for makeup
FACE_REGIONS = {
    "LIP_UPPER": [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 308, 415, 310, 312, 13, 82, 81, 80, 191, 78],
    "LIP_LOWER": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 402, 317, 14, 87, 178, 88, 95, 78, 61],
    "EYELINER_LEFT": [243, 112, 26, 22, 23, 24, 110, 25, 226, 130, 33, 7, 163, 144, 145, 153, 154, 155, 133, 243],
    "EYELINER_RIGHT": [463, 362, 382, 381, 380, 374, 373, 390, 249, 263, 359, 446, 255, 339, 254, 253, 252, 256, 341, 463],
    "EYESHADOW_LEFT": [226, 247, 30, 29, 27, 28, 56, 190, 243, 173, 157, 158, 159, 160, 161, 246, 33, 130, 226],
    "EYESHADOW_RIGHT": [463, 414, 286, 258, 257, 259, 260, 467, 446, 359, 263, 466, 388, 387, 386, 385, 384, 398, 362, 463],
    "EYEBROW_LEFT": [55, 107, 66, 105, 63, 70, 46, 53, 52, 65, 55],
    "EYEBROW_RIGHT": [285, 336, 296, 334, 293, 300, 276, 283, 295, 285],
    "BLUSH_LEFT": [116, 123, 147, 213, 192, 214, 210, 211, 32, 140, 116],
    "BLUSH_RIGHT": [345, 352, 376, 433, 416, 434, 430, 431, 262, 369, 345],
    # Face outline for foundation (simplified)
    "FACE": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152,
             148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10],
    # Eyes (for exclusion from foundation)
    "LEFT_EYE": [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7],
    "RIGHT_EYE": [362, 298, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382],
    # Simplified contour regions
    "CONTOUR_CHEEK_LEFT": [234, 93, 132, 123, 116, 143, 156, 70, 63, 105, 66, 107, 55, 193, 168, 234],
    "CONTOUR_CHEEK_RIGHT": [454, 323, 361, 352, 345, 372, 383, 300, 293, 334, 296, 336, 285, 417, 168, 454],
    "HIGHLIGHT_NOSE": [6, 197, 195, 5, 4, 1, 19, 94, 2, 164, 0, 11, 12, 13, 14, 15, 16, 17, 18, 200, 199, 175],
}


# ============ OPTIMIZED MAKEUP EFFECTS ============

def fast_blur(image, ksize=15):
    """Fast box blur - much faster than Gaussian blur."""
    return cv2.blur(image, (ksize, ksize))


def apply_foundation_fast(image, face_landmarks, strength=0.5, warmth=0):
    """
    FAST foundation effect - uses simple blur instead of bilateral filter.
    Much faster but slightly less edge-preserving.
    """
    h, w = image.shape[:2]
    coords = {}
    for idx, lm in enumerate(face_landmarks):
        coords[idx] = (int(lm.x * w), int(lm.y * h))
    
    # Create face mask
    face_indices = FACE_REGIONS.get("FACE", [])
    if not face_indices:
        return image
    
    try:
        face_points = np.array([coords[idx] for idx in face_indices], dtype=np.int32)
    except KeyError:
        return image
    
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [face_points], 255)
    
    # Exclude eyes and mouth (simplified - just the main regions)
    for region in ["LEFT_EYE", "RIGHT_EYE", "LIP_UPPER", "LIP_LOWER"]:
        indices = FACE_REGIONS.get(region, [])
        if indices:
            try:
                points = np.array([coords[idx] for idx in indices], dtype=np.int32)
                cv2.fillPoly(mask, [points], 0)
            except KeyError:
                pass
    
    # Fast feather with box blur
    mask = fast_blur(mask, 21)
    
    # Fast skin smoothing with box blur (much faster than bilateral)
    ksize = 9 + int(strength * 8)
    smoothed = cv2.blur(image, (ksize, ksize))
    
    # Apply warmth adjustment inline (avoid extra operations)
    if warmth != 0:
        if warmth > 0:
            smoothed = smoothed.astype(np.int16)
            smoothed[:, :, 2] = np.clip(smoothed[:, :, 2] + int(warmth * 0.5), 0, 255)
            smoothed[:, :, 1] = np.clip(smoothed[:, :, 1] + int(warmth * 0.3), 0, 255)
            smoothed = smoothed.astype(np.uint8)
        else:
            smoothed = smoothed.astype(np.int16)
            smoothed[:, :, 0] = np.clip(smoothed[:, :, 0] - int(warmth * 0.5), 0, 255)
            smoothed = smoothed.astype(np.uint8)
    
    # Fast blend using cv2.addWeighted where possible
    mask_norm = mask.astype(np.float32) / 255.0 * strength
    mask_3ch = np.dstack([mask_norm, mask_norm, mask_norm])
    
    # Use numpy operations optimized for speed
    result = (image.astype(np.float32) * (1 - mask_3ch) + smoothed.astype(np.float32) * mask_3ch).astype(np.uint8)
    
    return result


def apply_contour_fast(image, face_landmarks, shadow_strength=0.15, highlight_strength=0.1):
    """FAST contouring - simplified with fewer blurs."""
    h, w = image.shape[:2]
    coords = {}
    for idx, lm in enumerate(face_landmarks):
        coords[idx] = (int(lm.x * w), int(lm.y * h))
    
    # Create single overlay for all contour effects
    overlay = np.zeros_like(image)
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Contour (shadow) on cheekbones
    shadow_color = (60, 70, 85)
    for region in ["CONTOUR_CHEEK_LEFT", "CONTOUR_CHEEK_RIGHT"]:
        indices = FACE_REGIONS.get(region, [])
        if indices:
            try:
                points = np.array([coords[idx] for idx in indices], dtype=np.int32)
                cv2.fillPoly(overlay, [points], shadow_color)
                cv2.fillPoly(combined_mask, [points], 255)
            except KeyError:
                pass
    
    # Single blur pass for all contour regions
    overlay = fast_blur(overlay, 31)
    combined_mask = fast_blur(combined_mask, 31)
    
    # Fast blend
    mask_norm = combined_mask.astype(np.float32) / 255.0 * shadow_strength
    mask_3ch = np.dstack([mask_norm, mask_norm, mask_norm])
    result = (image.astype(np.float32) * (1 - mask_3ch) + overlay.astype(np.float32) * mask_3ch).astype(np.uint8)
    
    return result


def apply_lip_gloss_fast(image, face_landmarks, intensity=0.3):
    """FAST lip gloss - simplified highlight."""
    h, w = image.shape[:2]
    coords = {}
    for idx, lm in enumerate(face_landmarks):
        coords[idx] = (int(lm.x * w), int(lm.y * h))
    
    # Create combined lip mask
    lip_mask = np.zeros((h, w), dtype=np.uint8)
    lip_points_all = []
    for region in ["LIP_UPPER", "LIP_LOWER"]:
        indices = FACE_REGIONS.get(region, [])
        if indices:
            try:
                points = np.array([coords[idx] for idx in indices], dtype=np.int32)
                cv2.fillPoly(lip_mask, [points], 255)
                lip_points_all.extend([coords[idx] for idx in indices])
            except KeyError:
                pass
    
    if not lip_points_all:
        return image
    
    # Find lip center
    lip_points_arr = np.array(lip_points_all)
    center_x = int(lip_points_arr[:, 0].mean())
    center_y = int(lip_points_arr[:, 1].mean()) - 3
    lip_width = lip_points_arr[:, 0].max() - lip_points_arr[:, 0].min()
    
    # Create simple elliptical highlight
    gloss = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(gloss, (center_x, center_y), (int(lip_width * 0.2), int(lip_width * 0.08)), 
                0, 0, 360, 255, -1)
    gloss = fast_blur(gloss, 15)
    gloss = cv2.bitwise_and(gloss, lip_mask)
    
    # Fast blend
    gloss_norm = gloss.astype(np.float32) / 255.0 * intensity
    gloss_3ch = np.dstack([gloss_norm, gloss_norm, gloss_norm])
    highlight = np.full_like(image, 255, dtype=np.uint8)
    result = (image.astype(np.float32) * (1 - gloss_3ch) + highlight.astype(np.float32) * gloss_3ch).astype(np.uint8)
    
    return result


def apply_blush_fast(image, face_landmarks, color=(180, 140, 200), intensity=0.25):
    """FAST blush - single pass blur."""
    h, w = image.shape[:2]
    coords = {}
    for idx, lm in enumerate(face_landmarks):
        coords[idx] = (int(lm.x * w), int(lm.y * h))
    
    overlay = np.zeros_like(image)
    mask = np.zeros((h, w), dtype=np.uint8)
    
    for region in ["BLUSH_LEFT", "BLUSH_RIGHT"]:
        indices = FACE_REGIONS.get(region, [])
        if indices:
            try:
                points = np.array([coords[idx] for idx in indices], dtype=np.int32)
                cv2.fillPoly(overlay, [points], color)
                cv2.fillPoly(mask, [points], 255)
            except KeyError:
                pass
    
    # Single blur pass
    overlay = fast_blur(overlay, 41)
    mask = fast_blur(mask, 41)
    
    mask_norm = mask.astype(np.float32) / 255.0 * intensity
    mask_3ch = np.dstack([mask_norm, mask_norm, mask_norm])
    result = (image.astype(np.float32) * (1 - mask_3ch) + overlay.astype(np.float32) * mask_3ch).astype(np.uint8)
    
    return result


# Makeup looks - optimized versions with reduced foundation strength for speed
MAKEUP_LOOKS = {
    'none': {
        'name': 'No Makeup',
        'features': {},
    },
    'natural': {
        'name': 'Natural Beauty',
        'foundation': 0.2,  # Reduced for speed
        'features': {
            'LIP_UPPER': {'color': (140, 150, 195), 'alpha': 0.25},
            'LIP_LOWER': {'color': (140, 150, 195), 'alpha': 0.25},
            'EYEBROW_LEFT': {'color': (60, 70, 80), 'alpha': 0.15},
            'EYEBROW_RIGHT': {'color': (60, 70, 80), 'alpha': 0.15},
        },
        'lip_gloss': 0.2,
    },
    'flawless': {
        'name': 'Flawless Skin',
        'foundation': 0.4,  # Reduced for speed
        'contour': True,
        'features': {
            'EYEBROW_LEFT': {'color': (50, 60, 70), 'alpha': 0.2},
            'EYEBROW_RIGHT': {'color': (50, 60, 70), 'alpha': 0.2},
        },
        'blush_glow': {'color': (180, 160, 200), 'intensity': 0.2},
    },
    'glamour': {
        'name': 'Red Carpet',
        'foundation': 0.3,
        'contour': True,
        'features': {
            'LIP_UPPER': {'color': (50, 50, 180), 'alpha': 0.55},
            'LIP_LOWER': {'color': (50, 50, 180), 'alpha': 0.55},
            'EYELINER_LEFT': {'color': (30, 30, 30), 'alpha': 0.7},
            'EYELINER_RIGHT': {'color': (30, 30, 30), 'alpha': 0.7},
            'EYESHADOW_LEFT': {'color': (120, 80, 150), 'alpha': 0.4},
            'EYESHADOW_RIGHT': {'color': (120, 80, 150), 'alpha': 0.4},
        },
        'blush_glow': {'color': (180, 140, 200), 'intensity': 0.25},
        'lip_gloss': 0.35,
    },
    'bold': {
        'name': 'Bold Red',
        'foundation': 0.3,
        'features': {
            'LIP_UPPER': {'color': (0, 0, 200), 'alpha': 0.65},
            'LIP_LOWER': {'color': (0, 0, 200), 'alpha': 0.65},
            'EYELINER_LEFT': {'color': (0, 0, 0), 'alpha': 0.75},
            'EYELINER_RIGHT': {'color': (0, 0, 0), 'alpha': 0.75},
            'EYESHADOW_LEFT': {'color': (80, 60, 120), 'alpha': 0.4},
            'EYESHADOW_RIGHT': {'color': (80, 60, 120), 'alpha': 0.4},
        },
        'lip_gloss': 0.4,
    },
    'smoky': {
        'name': 'Smoky Night',
        'foundation': 0.3,
        'features': {
            'LIP_UPPER': {'color': (120, 110, 150), 'alpha': 0.4},
            'LIP_LOWER': {'color': (120, 110, 150), 'alpha': 0.4},
            'EYELINER_LEFT': {'color': (20, 20, 20), 'alpha': 0.85},
            'EYELINER_RIGHT': {'color': (20, 20, 20), 'alpha': 0.85},
            'EYESHADOW_LEFT': {'color': (50, 40, 60), 'alpha': 0.55},
            'EYESHADOW_RIGHT': {'color': (50, 40, 60), 'alpha': 0.55},
        },
    },
    'coral': {
        'name': 'Coral Summer',
        'foundation': 0.25,
        'foundation_warmth': 15,
        'features': {
            'LIP_UPPER': {'color': (100, 130, 230), 'alpha': 0.5},
            'LIP_LOWER': {'color': (100, 130, 230), 'alpha': 0.5},
            'EYESHADOW_LEFT': {'color': (150, 180, 220), 'alpha': 0.3},
            'EYESHADOW_RIGHT': {'color': (150, 180, 220), 'alpha': 0.3},
        },
        'blush_glow': {'color': (130, 160, 230), 'intensity': 0.3},
        'lip_gloss': 0.35,
    },
    'gothic': {
        'name': 'Gothic Queen',
        'foundation': 0.35,
        'foundation_warmth': -20,
        'features': {
            'LIP_UPPER': {'color': (30, 0, 80), 'alpha': 0.75},
            'LIP_LOWER': {'color': (30, 0, 80), 'alpha': 0.75},
            'EYELINER_LEFT': {'color': (0, 0, 0), 'alpha': 0.9},
            'EYELINER_RIGHT': {'color': (0, 0, 0), 'alpha': 0.9},
            'EYESHADOW_LEFT': {'color': (40, 20, 50), 'alpha': 0.6},
            'EYESHADOW_RIGHT': {'color': (40, 20, 50), 'alpha': 0.6},
            'EYEBROW_LEFT': {'color': (15, 15, 15), 'alpha': 0.4},
            'EYEBROW_RIGHT': {'color': (15, 15, 15), 'alpha': 0.4},
        },
    },
    'fairy': {
        'name': 'Fairy Princess',
        'foundation': 0.25,
        'features': {
            'LIP_UPPER': {'color': (200, 150, 220), 'alpha': 0.35},
            'LIP_LOWER': {'color': (200, 150, 220), 'alpha': 0.35},
            'EYESHADOW_LEFT': {'color': (255, 200, 230), 'alpha': 0.35},
            'EYESHADOW_RIGHT': {'color': (255, 200, 230), 'alpha': 0.35},
        },
        'blush_glow': {'color': (230, 190, 240), 'intensity': 0.3},
        'lip_gloss': 0.4,
    },
    'bronze': {
        'name': 'Bronze Goddess',
        'foundation': 0.3,
        'foundation_warmth': 25,
        'contour': True,
        'features': {
            'LIP_UPPER': {'color': (80, 120, 180), 'alpha': 0.45},
            'LIP_LOWER': {'color': (80, 120, 180), 'alpha': 0.45},
            'EYESHADOW_LEFT': {'color': (50, 120, 190), 'alpha': 0.4},
            'EYESHADOW_RIGHT': {'color': (50, 120, 190), 'alpha': 0.4},
            'EYELINER_LEFT': {'color': (40, 80, 120), 'alpha': 0.5},
            'EYELINER_RIGHT': {'color': (40, 80, 120), 'alpha': 0.5},
        },
        'blush_glow': {'color': (100, 150, 200), 'intensity': 0.25},
        'lip_gloss': 0.3,
    },
    'dewy': {
        'name': 'Dewy Fresh',
        'foundation': 0.3,
        'features': {
            'LIP_UPPER': {'color': (160, 140, 190), 'alpha': 0.3},
            'LIP_LOWER': {'color': (160, 140, 190), 'alpha': 0.3},
        },
        'blush_glow': {'color': (200, 170, 210), 'intensity': 0.35},
        'lip_gloss': 0.5,
    },
}

MAKEUP_KEYS = list(MAKEUP_LOOKS.keys())


def get_landmark_coords(face_landmarks, image_shape):
    """Extract pixel coordinates for all face landmarks."""
    h, w = image_shape[:2]
    coords = {}
    for idx, lm in enumerate(face_landmarks):
        coords[idx] = (int(lm.x * w), int(lm.y * h))
    return coords


def apply_makeup_fast(image, face_landmarks, makeup_look):
    """
    FAST makeup application - optimized for real-time performance.
    Combines multiple effects into fewer passes.
    """
    # Check if this is the "no makeup" look
    if not makeup_look.get('features') and not makeup_look.get('foundation'):
        return image
    
    result = image
    
    # 1. Apply foundation (skin smoothing) - skip if performance critical
    foundation_strength = makeup_look.get('foundation', 0)
    if foundation_strength > 0:
        warmth = makeup_look.get('foundation_warmth', 0)
        result = apply_foundation_fast(result, face_landmarks, strength=foundation_strength, warmth=warmth)
    
    # 2. Apply contouring
    if makeup_look.get('contour', False):
        result = apply_contour_fast(result, face_landmarks)
    
    # 3. Apply color makeup (all features in single pass)
    features = makeup_look.get('features', {})
    if features:
        h, w = result.shape[:2]
        coords = get_landmark_coords(face_landmarks, result.shape)
        
        # Create single overlay for all features
        overlay = result.copy()
        combined_mask = np.zeros((h, w), dtype=np.float32)
        
        for region_name, style in features.items():
            if region_name not in FACE_REGIONS:
                continue
            
            region_indices = FACE_REGIONS[region_name]
            color = style['color']
            alpha = style.get('alpha', 0.5)
            
            try:
                points = np.array([coords[idx] for idx in region_indices], dtype=np.int32)
                cv2.fillPoly(overlay, [points], color)
                
                # Build combined mask with per-feature alpha
                temp_mask = np.zeros((h, w), dtype=np.float32)
                cv2.fillPoly(temp_mask, [points], alpha)
                combined_mask = np.maximum(combined_mask, temp_mask)
            except KeyError:
                continue
        
        # Single blur pass for all features
        overlay = fast_blur(overlay, 5)
        combined_mask = cv2.blur(combined_mask, (11, 11))
        
        # Apply combined mask
        mask_3ch = np.dstack([combined_mask, combined_mask, combined_mask])
        result = (result.astype(np.float32) * (1 - mask_3ch) + overlay.astype(np.float32) * mask_3ch).astype(np.uint8)
    
    # 4. Apply blush
    blush_config = makeup_look.get('blush_glow')
    if blush_config:
        color = blush_config.get('color', (180, 140, 200))
        intensity = blush_config.get('intensity', 0.25)
        result = apply_blush_fast(result, face_landmarks, color=color, intensity=intensity)
    
    # 5. Apply lip gloss
    lip_gloss_intensity = makeup_look.get('lip_gloss', 0)
    if lip_gloss_intensity > 0:
        result = apply_lip_gloss_fast(result, face_landmarks, intensity=lip_gloss_intensity)
    
    return result


# ============ 3D MODEL LOADER ============

def load_obj(filepath):
    """Load a simple OBJ file and return vertices and faces."""
    vertices = []
    faces = []
    
    if not os.path.exists(filepath):
        print(f"Warning: Model file not found: {filepath}")
        return np.array([]), []
    
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith('f '):
                parts = line.strip().split()
                face = []
                for p in parts[1:]:
                    idx = int(p.split('/')[0]) - 1
                    face.append(idx)
                faces.append(face)
    
    return np.array(vertices, dtype=np.float32), faces


# ============ FACE MESH BASED 3D TRANSFORM ============

def get_face_transform(face_landmarks, image_shape):
    """Extract 3D transformation from face landmarks."""
    h, w = image_shape[:2]
    
    FOREHEAD = 10
    CHIN = 152
    LEFT_EYE = 263
    RIGHT_EYE = 33
    
    def get_3d(idx):
        lm = face_landmarks[idx]
        return np.array([lm.x * w, lm.y * h, lm.z * w])
    
    forehead = get_3d(FOREHEAD)
    chin = get_3d(CHIN)
    left_eye = get_3d(LEFT_EYE)
    right_eye = get_3d(RIGHT_EYE)
    
    up = forehead - chin
    up = up / (np.linalg.norm(up) + 1e-6)
    
    right = right_eye - left_eye
    right = right / (np.linalg.norm(right) + 1e-6)
    
    forward = np.cross(right, up)
    forward = forward / (np.linalg.norm(forward) + 1e-6)
    
    up = np.cross(forward, right)
    up = up / (np.linalg.norm(up) + 1e-6)
    
    face_height = np.linalg.norm(forehead - chin)
    face_width = np.linalg.norm(right_eye - left_eye)
    
    head_top = forehead + up * face_height * 0.3
    
    return {
        'origin': head_top,
        'right': right,
        'up': up,
        'forward': forward,
        'face_height': face_height,
        'face_width': face_width,
    }


# ============ 3D MODEL RENDERING ============

def project_model_from_face(vertices, face_transform, scale_factor, model_offset=(0, 0, 0)):
    """Project 3D model using face coordinate system."""
    if len(vertices) == 0:
        return np.array([])
    
    origin = face_transform['origin']
    right = face_transform['right']
    up = face_transform['up']
    forward = face_transform['forward']
    face_height = face_transform['face_height']
    
    scale = (face_height / 120.0) * scale_factor
    
    offset_right, offset_up, offset_forward = model_offset
    offset_world = (right * offset_right + up * offset_up + forward * offset_forward) * scale
    adjusted_origin = origin + offset_world
    
    rotation_matrix = np.column_stack([right, -up, forward])
    
    # Vectorized projection for speed
    scaled = vertices * scale
    rotated = (rotation_matrix @ scaled.T).T
    world_pos = adjusted_origin + rotated
    
    return world_pos[:, :2].astype(np.int32)


def render_model_wireframe(image, projected_points, faces, color=(0, 255, 255), thickness=2):
    """Render 3D model as wireframe."""
    if len(projected_points) == 0:
        return
    
    for face in faces:
        pts = [projected_points[i] for i in face if i < len(projected_points)]
        if len(pts) >= 2:
            for i in range(len(pts)):
                pt1 = tuple(pts[i])
                pt2 = tuple(pts[(i + 1) % len(pts)])
                cv2.line(image, pt1, pt2, color, thickness)


def render_model_filled(image, projected_points, faces, color=(0, 255, 255), alpha=0.6):
    """Render 3D model with filled polygons."""
    if len(projected_points) == 0:
        return
    
    overlay = image.copy()
    
    for face in faces:
        pts = np.array([projected_points[i] for i in face if i < len(projected_points)])
        if len(pts) >= 3:
            cv2.fillPoly(overlay, [pts], color)
    
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)


# ============ BUILT-IN 3D MODELS ============

def create_crown_model():
    """Create a simple crown 3D model."""
    vertices = []
    faces = []
    
    n_points = 8
    radius = 30
    
    for i in range(n_points):
        angle = 2 * np.pi * i / n_points
        vertices.append([radius * np.cos(angle), 0, radius * np.sin(angle)])
    
    peak_height = -40
    for i in range(n_points):
        angle = 2 * np.pi * (i + 0.5) / n_points
        vertices.append([radius * 0.9 * np.cos(angle), peak_height, radius * 0.9 * np.sin(angle)])
    
    for i in range(n_points):
        angle = 2 * np.pi * i / n_points
        vertices.append([radius * 0.7 * np.cos(angle), peak_height * 0.5, radius * 0.7 * np.sin(angle)])
    
    for i in range(n_points):
        next_i = (i + 1) % n_points
        faces.append([i, next_i, n_points + i])
        faces.append([next_i, n_points + next_i, n_points + i])
        faces.append([n_points + i, n_points + next_i, 2*n_points + i])
        faces.append([n_points + next_i, 2*n_points + next_i, 2*n_points + i])
    
    return np.array(vertices, dtype=np.float32), faces


def create_hat_model():
    """Create a simple top hat 3D model."""
    vertices = []
    faces = []
    
    n_points = 12  # Reduced from 16 for speed
    brim_radius = 45
    top_radius = 25
    hat_height = -50
    
    for i in range(n_points):
        angle = 2 * np.pi * i / n_points
        vertices.append([brim_radius * np.cos(angle), 0, brim_radius * np.sin(angle)])
    
    for i in range(n_points):
        angle = 2 * np.pi * i / n_points
        vertices.append([top_radius * np.cos(angle), 0, top_radius * np.sin(angle)])
    
    for i in range(n_points):
        angle = 2 * np.pi * i / n_points
        vertices.append([top_radius * np.cos(angle), hat_height, top_radius * np.sin(angle)])
    
    top_center_idx = len(vertices)
    vertices.append([0, hat_height, 0])
    
    for i in range(n_points):
        next_i = (i + 1) % n_points
        faces.append([i, next_i, n_points + next_i, n_points + i])
    
    for i in range(n_points):
        next_i = (i + 1) % n_points
        faces.append([n_points + i, n_points + next_i, 2*n_points + next_i, 2*n_points + i])
    
    for i in range(n_points):
        next_i = (i + 1) % n_points
        faces.append([2*n_points + i, 2*n_points + next_i, top_center_idx])
    
    return np.array(vertices, dtype=np.float32), faces


def create_horns_model():
    """Create devil horns 3D model."""
    vertices = []
    faces = []
    
    def add_horn(x_offset, curve_dir):
        base_idx = len(vertices)
        n_segments = 4  # Reduced from 6 for speed
        base_radius = 8
        
        for seg in range(n_segments + 1):
            t = seg / n_segments
            y = -t * 50
            x = x_offset + curve_dir * t * t * 20
            z = t * 10
            radius = base_radius * (1 - t * 0.8)
            
            for i in range(4):  # Reduced from 6 for speed
                angle = 2 * np.pi * i / 4
                vertices.append([x + radius * np.cos(angle), y, z + radius * np.sin(angle)])
        
        for seg in range(n_segments):
            for i in range(4):
                next_i = (i + 1) % 4
                idx = base_idx + seg * 4
                faces.append([idx + i, idx + next_i, idx + 4 + next_i, idx + 4 + i])
    
    add_horn(-30, -1)
    add_horn(30, 1)
    
    return np.array(vertices, dtype=np.float32), faces


def create_halo_model():
    """Create a halo/angel ring 3D model."""
    vertices = []
    faces = []
    
    n_points = 16  # Reduced from 24 for speed
    outer_radius = 40
    inner_radius = 32
    ring_height = -50
    thickness = 3
    
    for i in range(n_points):
        angle = 2 * np.pi * i / n_points
        vertices.append([outer_radius * np.cos(angle), ring_height - thickness, outer_radius * np.sin(angle)])
    
    for i in range(n_points):
        angle = 2 * np.pi * i / n_points
        vertices.append([inner_radius * np.cos(angle), ring_height - thickness, inner_radius * np.sin(angle)])
    
    for i in range(n_points):
        angle = 2 * np.pi * i / n_points
        vertices.append([outer_radius * np.cos(angle), ring_height + thickness, outer_radius * np.sin(angle)])
    
    for i in range(n_points):
        angle = 2 * np.pi * i / n_points
        vertices.append([inner_radius * np.cos(angle), ring_height + thickness, inner_radius * np.sin(angle)])
    
    for i in range(n_points):
        next_i = (i + 1) % n_points
        faces.append([i, next_i, n_points + next_i, n_points + i])
        faces.append([2*n_points + i, 3*n_points + i, 3*n_points + next_i, 2*n_points + next_i])
        faces.append([i, 2*n_points + i, 2*n_points + next_i, next_i])
        faces.append([n_points + i, n_points + next_i, 3*n_points + next_i, 3*n_points + i])
    
    return np.array(vertices, dtype=np.float32), faces


def create_party_hat_model():
    """Create a party/cone hat 3D model."""
    vertices = []
    faces = []
    
    n_points = 8  # Reduced from 12 for speed
    base_radius = 30
    hat_height = -70
    
    for i in range(n_points):
        angle = 2 * np.pi * i / n_points
        vertices.append([base_radius * np.cos(angle), 0, base_radius * np.sin(angle)])
    
    tip_idx = len(vertices)
    vertices.append([0, hat_height, 0])
    
    for i in range(n_points):
        next_i = (i + 1) % n_points
        faces.append([i, next_i, tip_idx])
    
    base_center_idx = len(vertices)
    vertices.append([0, 0, 0])
    for i in range(n_points):
        next_i = (i + 1) % n_points
        faces.append([next_i, i, base_center_idx])
    
    return np.array(vertices, dtype=np.float32), faces


# ============ MAIN APPLICATION ============

MODELS = {
    'crown': {
        'create': create_crown_model,
        'color': (0, 215, 255),
        'name': 'Golden Crown',
        'offset': (0, 5, 60),
    },
    'hat': {
        'create': create_hat_model,
        'color': (40, 40, 40),
        'name': 'Top Hat',
        'offset': (0, 15, 60),
    },
    'horns': {
        'create': create_horns_model,
        'color': (0, 0, 180),
        'name': 'Devil Horns',
        'offset': (0, -5, 30),
    },
    'halo': {
        'create': create_halo_model,
        'color': (0, 255, 255),
        'name': 'Angel Halo',
        'offset': (0, 20, 60),
    },
    'party': {
        'create': create_party_hat_model,
        'color': (255, 0, 128),
        'name': 'Party Hat',
        'offset': (0, 5, 60),
    },
    'wings': {
        'create': lambda: load_obj('models/wings.obj'),
        'color': (255, 255, 255),
        'name': 'Wings',
        'offset': (0, 150, 80),
    },
}

MODEL_KEYS = list(MODELS.keys())


def draw_face_mesh_landmarks(image, face_landmarks):
    """Draw face mesh landmarks (simplified for speed)."""
    h, w = image.shape[:2]
    # Draw every 5th landmark for speed
    for i, lm in enumerate(face_landmarks):
        if i % 5 == 0:
            cv2.circle(image, (int(lm.x * w), int(lm.y * h)), 1, (0, 255, 0), -1)


def main():
    ensure_model_exists()
    
    current_model_idx = 0
    show_face_mesh = False
    render_filled = True
    model_scale = 1.0
    current_makeup_idx = 0
    
    def get_current_model():
        key = MODEL_KEYS[current_model_idx]
        model_data = MODELS[key]
        vertices, faces = model_data['create']()
        offset = model_data.get('offset', (0, 0, 0))
        return vertices, faces, model_data['color'], model_data['name'], offset
    
    def get_current_makeup():
        key = MAKEUP_KEYS[current_makeup_idx]
        return MAKEUP_LOOKS[key]
    
    vertices, faces, model_color, model_name, model_offset = get_current_model()
    current_makeup = get_current_makeup()
    
    cap = cv2.VideoCapture(0)
    
    # Use lower resolution for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cam_fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    
    print(f"Camera resolution: {cam_width}x{cam_height} @ {cam_fps}fps")
    print(f"Processing resolution: {PROCESS_WIDTH}x{PROCESS_HEIGHT}")
    
    print("\n=== 3D Head Model AR Demo (OPTIMIZED) ===")
    print("Controls:")
    print("  [N/P] - Next/Previous hat model")
    print("  [</,] - Previous makeup look")
    print("  [>/.]- Next makeup look")
    print("  [M] - Toggle face mesh overlay")
    print("  [F] - Toggle filled/wireframe rendering")
    print("  [V] - Toggle virtual camera")
    print("  [+/-] - Adjust model scale")
    print("  [ESC] - Exit")
    print("=" * 30 + "\n")
    
    vcam = None
    vcam_enabled = False
    if VIRTUALCAM_AVAILABLE:
        try:
            vcam = pyvirtualcam.Camera(width=cam_width, height=cam_height, fps=cam_fps, fmt=pyvirtualcam.PixelFormat.BGR)
            vcam_enabled = True
            print(f"Virtual camera started: {vcam.device}")
        except Exception as e:
            print(f"Could not start virtual camera: {e}")
            vcam = None
    
    # Create FaceLandmarker with VIDEO mode for tracking
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,  # VIDEO mode has tracking
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False
    )
    
    face_landmarker = vision.FaceLandmarker.create_from_options(options)
    
    # FPS tracking
    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0
    frame_timestamp = 0
    
    try:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue
            
            image = cv2.flip(image, 1)
            
            # Downscale for faster processing
            h, w = image.shape[:2]
            scale_x = PROCESS_WIDTH / w
            scale_y = PROCESS_HEIGHT / h
            small_image = cv2.resize(image, (PROCESS_WIDTH, PROCESS_HEIGHT))
            
            # Process face mesh using VIDEO mode
            image_rgb = cv2.cvtColor(small_image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            frame_timestamp += 33  # ~30fps
            results = face_landmarker.detect_for_video(mp_image, frame_timestamp)
            
            if results.face_landmarks:
                for face_landmarks in results.face_landmarks:
                    # Apply makeup on small image
                    small_image = apply_makeup_fast(small_image, face_landmarks, current_makeup)
                    
                    if show_face_mesh:
                        draw_face_mesh_landmarks(small_image, face_landmarks)
                    
                    # Scale landmarks for full-size rendering
                    face_transform = get_face_transform(face_landmarks, small_image.shape)
                    
                    # Scale transform origin to full size
                    face_transform['origin'] = face_transform['origin'] / np.array([scale_x, scale_y, scale_x])
                    face_transform['face_height'] = face_transform['face_height'] / scale_y
                    
                    projected = project_model_from_face(vertices, face_transform, model_scale, model_offset)
            
            # Upscale processed image back to original size
            image = cv2.resize(small_image, (w, h))
            
            # Render 3D model on full-size image
            if results.face_landmarks and len(projected) > 0:
                if render_filled:
                    render_model_filled(image, projected, faces, model_color, alpha=0.7)
                render_model_wireframe(image, projected, faces, model_color, thickness=2)
            
            # Virtual camera output
            if vcam is not None and vcam_enabled:
                vcam.send(image)
                vcam.sleep_until_next_frame()
            
            # FPS calculation
            fps_counter += 1
            elapsed = time.time() - fps_start_time
            if elapsed >= 1.0:
                current_fps = fps_counter / elapsed
                fps_counter = 0
                fps_start_time = time.time()
            
            # UI overlay
            preview = image.copy()
            cv2.rectangle(preview, (10, 10), (320, 130), (0, 0, 0), -1)
            cv2.rectangle(preview, (10, 10), (320, 130), model_color, 2)
            cv2.putText(preview, f"FPS: {current_fps:.1f}", (20, 32), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
            cv2.putText(preview, f"Hat: {model_name}", (20, 52), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(preview, f"Makeup: {current_makeup['name']}", (20, 72), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 255), 1)
            cv2.putText(preview, f"Scale: {model_scale:.1f}x", (20, 92), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            vcam_status = "ON" if (vcam is not None and vcam_enabled) else "OFF"
            cv2.putText(preview, f"[N/P] Hat | [</>] Makeup | [V]Cam:{vcam_status}", (20, 112), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
            cv2.putText(preview, f"[F] Fill | [M] Mesh | [+/-] Scale", (20, 127), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
            
            cv2.imshow('3D Head Model AR', preview)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:
                break
            elif key == ord('n') or key == ord('N'):
                current_model_idx = (current_model_idx + 1) % len(MODEL_KEYS)
                vertices, faces, model_color, model_name, model_offset = get_current_model()
                print(f"Hat: {model_name}")
            elif key == ord('p') or key == ord('P'):
                current_model_idx = (current_model_idx - 1) % len(MODEL_KEYS)
                vertices, faces, model_color, model_name, model_offset = get_current_model()
                print(f"Hat: {model_name}")
            elif key == ord('.') or key == ord('>'):
                current_makeup_idx = (current_makeup_idx + 1) % len(MAKEUP_KEYS)
                current_makeup = get_current_makeup()
                print(f"Makeup: {current_makeup['name']}")
            elif key == ord(',') or key == ord('<'):
                current_makeup_idx = (current_makeup_idx - 1) % len(MAKEUP_KEYS)
                current_makeup = get_current_makeup()
                print(f"Makeup: {current_makeup['name']}")
            elif key == ord('m') or key == ord('M'):
                show_face_mesh = not show_face_mesh
                print(f"Face mesh: {'ON' if show_face_mesh else 'OFF'}")
            elif key == ord('f') or key == ord('F'):
                render_filled = not render_filled
                print(f"Filled rendering: {'ON' if render_filled else 'OFF'}")
            elif key == ord('+') or key == ord('='):
                model_scale = min(3.0, model_scale + 0.1)
                print(f"Scale: {model_scale:.1f}x")
            elif key == ord('-') or key == ord('_'):
                model_scale = max(0.3, model_scale - 0.1)
                print(f"Scale: {model_scale:.1f}x")
            elif key == ord('v') or key == ord('V'):
                if vcam is not None:
                    vcam_enabled = not vcam_enabled
                    print(f"Virtual camera: {'ON' if vcam_enabled else 'OFF'}")
                else:
                    print("Virtual camera not available")
    
    finally:
        face_landmarker.close()
        cap.release()
        if vcam is not None:
            vcam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
