import cv2
import mediapipe as mp
import numpy as np
import os
import urllib.request

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
    # Face outline for foundation
    "FACE": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152,
             148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10],
    # Eyes (for exclusion from foundation)
    "LEFT_EYE": [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7],
    "RIGHT_EYE": [362, 298, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382],
    # Contour regions
    "CONTOUR_CHEEK_LEFT": [234, 93, 132, 123, 116, 143, 156, 70, 63, 105, 66, 107, 55, 193, 168, 234],
    "CONTOUR_CHEEK_RIGHT": [454, 323, 361, 352, 345, 372, 383, 300, 293, 334, 296, 336, 285, 417, 168, 454],
    "HIGHLIGHT_NOSE": [6, 197, 195, 5, 4, 1, 19, 94, 2, 164, 0, 11, 12, 13, 14, 15, 16, 17, 18, 200, 199, 175],
    "HIGHLIGHT_FOREHEAD": [10, 109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 58, 172, 138, 213, 147, 123, 116],
}


# ============ ADVANCED MAKEUP EFFECTS ============

def apply_foundation(image, face_landmarks, strength=0.5, warmth=0):
    """
    Apply foundation effect - smooths skin while preserving features.
    Uses bilateral filtering for edge-preserving smoothing.
    
    strength: 0-1, how much smoothing
    warmth: -50 to 50, negative=cooler, positive=warmer tone
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
    
    # Exclude eyes, mouth, and eyebrows from smoothing
    exclude_regions = ["LEFT_EYE", "RIGHT_EYE", "LIP_UPPER", "LIP_LOWER", "EYEBROW_LEFT", "EYEBROW_RIGHT"]
    for region in exclude_regions:
        indices = FACE_REGIONS.get(region, [])
        if indices:
            try:
                points = np.array([coords[idx] for idx in indices], dtype=np.int32)
                cv2.fillPoly(mask, [points], 0)
            except KeyError:
                pass
    
    # Feather the mask edges
    mask = cv2.GaussianBlur(mask, (31, 31), 15)
    
    # Apply bilateral filter for skin smoothing (preserves edges like nose, lips)
    # Higher values = more smoothing
    d = 9 + int(strength * 10)  # Filter size
    sigma_color = 50 + int(strength * 100)  # Color similarity
    sigma_space = 50 + int(strength * 100)  # Spatial proximity
    smoothed = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    
    # Apply second pass for stronger effect
    if strength > 0.5:
        smoothed = cv2.bilateralFilter(smoothed, d, sigma_color // 2, sigma_space // 2)
    
    # Adjust skin tone warmth
    if warmth != 0:
        smoothed = smoothed.astype(np.float32)
        if warmth > 0:
            # Warmer - add red/yellow
            smoothed[:, :, 2] = np.clip(smoothed[:, :, 2] + warmth * 0.5, 0, 255)  # Red
            smoothed[:, :, 1] = np.clip(smoothed[:, :, 1] + warmth * 0.3, 0, 255)  # Green
        else:
            # Cooler - add blue
            smoothed[:, :, 0] = np.clip(smoothed[:, :, 0] - warmth * 0.5, 0, 255)  # Blue
        smoothed = smoothed.astype(np.uint8)
    
    # Blend smoothed skin with original using mask
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(float) / 255.0
    result = (image * (1 - mask_3ch * strength) + smoothed * mask_3ch * strength).astype(np.uint8)
    
    return result


def apply_contour(image, face_landmarks, shadow_strength=0.15, highlight_strength=0.1):
    """
    Apply contouring - darkens hollows, highlights high points.
    Creates a sculpted, dimensional look.
    """
    h, w = image.shape[:2]
    coords = {}
    for idx, lm in enumerate(face_landmarks):
        coords[idx] = (int(lm.x * w), int(lm.y * h))
    
    result = image.copy()
    
    # Contour (shadow) on cheekbones
    shadow_color = (60, 70, 85)  # Warm brown shadow
    for region in ["CONTOUR_CHEEK_LEFT", "CONTOUR_CHEEK_RIGHT"]:
        indices = FACE_REGIONS.get(region, [])
        if indices:
            try:
                points = np.array([coords[idx] for idx in indices], dtype=np.int32)
                overlay = result.copy()
                cv2.fillPoly(overlay, [points], shadow_color)
                overlay = cv2.GaussianBlur(overlay, (45, 45), 20)
                
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [points], 255)
                mask = cv2.GaussianBlur(mask, (45, 45), 20)
                mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(float) / 255.0
                
                result = (result * (1 - mask_3ch * shadow_strength) + overlay * mask_3ch * shadow_strength).astype(np.uint8)
            except KeyError:
                pass
    
    # Highlight on nose bridge and forehead
    highlight_color = (220, 215, 210)  # Soft white highlight
    for region in ["HIGHLIGHT_NOSE"]:
        indices = FACE_REGIONS.get(region, [])
        if indices:
            try:
                points = np.array([coords[idx] for idx in indices], dtype=np.int32)
                overlay = result.copy()
                cv2.fillPoly(overlay, [points], highlight_color)
                overlay = cv2.GaussianBlur(overlay, (35, 35), 15)
                
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [points], 255)
                mask = cv2.GaussianBlur(mask, (35, 35), 15)
                mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(float) / 255.0
                
                result = (result * (1 - mask_3ch * highlight_strength) + overlay * mask_3ch * highlight_strength).astype(np.uint8)
            except KeyError:
                pass
    
    return result


def apply_lip_gloss(image, face_landmarks, intensity=0.3):
    """Add glossy/shiny effect to lips with specular highlights."""
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
    
    # Find lip center for highlight placement
    lip_points_arr = np.array(lip_points_all)
    center_x = int(lip_points_arr[:, 0].mean())
    center_y = int(lip_points_arr[:, 1].mean()) - 5  # Slightly above center
    
    # Calculate lip width for scaling
    lip_width = lip_points_arr[:, 0].max() - lip_points_arr[:, 0].min()
    
    # Create glossy highlight (elliptical bright spot)
    gloss = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(gloss, (center_x, center_y), (int(lip_width * 0.25), int(lip_width * 0.1)), 
                0, 0, 360, 255, -1)
    gloss = cv2.GaussianBlur(gloss, (21, 21), 8)
    
    # Apply only within lip area
    gloss = cv2.bitwise_and(gloss, lip_mask)
    gloss = cv2.GaussianBlur(gloss, (15, 15), 5)
    
    # Add white specular highlight
    result = image.copy()
    gloss_3ch = cv2.cvtColor(gloss, cv2.COLOR_GRAY2BGR).astype(float) / 255.0
    highlight = np.full_like(image, (255, 255, 255), dtype=np.uint8)
    result = (result * (1 - gloss_3ch * intensity) + highlight * gloss_3ch * intensity).astype(np.uint8)
    
    return result


def apply_blush_glow(image, face_landmarks, color=(180, 140, 200), intensity=0.25):
    """Apply a soft, diffused blush with natural gradient."""
    h, w = image.shape[:2]
    coords = {}
    for idx, lm in enumerate(face_landmarks):
        coords[idx] = (int(lm.x * w), int(lm.y * h))
    
    result = image.copy()
    overlay = image.copy()
    
    for region in ["BLUSH_LEFT", "BLUSH_RIGHT"]:
        indices = FACE_REGIONS.get(region, [])
        if indices:
            try:
                points = np.array([coords[idx] for idx in indices], dtype=np.int32)
                
                # Draw blush with gradient
                cv2.fillPoly(overlay, [points], color)
            except KeyError:
                pass
    
    # Heavy blur for soft, natural gradient
    overlay = cv2.GaussianBlur(overlay, (61, 61), 30)
    
    # Create mask for blush areas
    mask = np.zeros((h, w), dtype=np.uint8)
    for region in ["BLUSH_LEFT", "BLUSH_RIGHT"]:
        indices = FACE_REGIONS.get(region, [])
        if indices:
            try:
                points = np.array([coords[idx] for idx in indices], dtype=np.int32)
                cv2.fillPoly(mask, [points], 255)
            except KeyError:
                pass
    
    mask = cv2.GaussianBlur(mask, (61, 61), 30)
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(float) / 255.0
    
    result = (result * (1 - mask_3ch * intensity) + overlay * mask_3ch * intensity).astype(np.uint8)
    
    return result

# Makeup looks - each defines colors for different regions
# Colors are in BGR format, with optional alpha (opacity)
# Advanced options: foundation, contour, lip_gloss, blush_glow
MAKEUP_LOOKS = {
    'none': {
        'name': 'No Makeup',
        'features': {},
    },
    'natural': {
        'name': 'Natural Beauty',
        'foundation': 0.3,  # Light skin smoothing
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
        'foundation': 0.6,  # Strong skin smoothing
        'contour': True,
        'features': {
            'EYEBROW_LEFT': {'color': (50, 60, 70), 'alpha': 0.2},
            'EYEBROW_RIGHT': {'color': (50, 60, 70), 'alpha': 0.2},
        },
        'blush_glow': {'color': (180, 160, 200), 'intensity': 0.2},
    },
    'glamour': {
        'name': 'Red Carpet',
        'foundation': 0.5,
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
        'foundation': 0.45,
        'contour': True,
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
        'foundation': 0.5,
        'contour': True,
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
        'foundation': 0.35,
        'foundation_warmth': 15,  # Warmer skin tone
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
        'foundation': 0.55,
        'foundation_warmth': -20,  # Cooler, paler skin
        'contour': True,
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
        'foundation': 0.4,
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
        'foundation': 0.45,
        'foundation_warmth': 25,  # Warm, sun-kissed
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
        'foundation': 0.5,
        'features': {
            'LIP_UPPER': {'color': (160, 140, 190), 'alpha': 0.3},
            'LIP_LOWER': {'color': (160, 140, 190), 'alpha': 0.3},
        },
        'blush_glow': {'color': (200, 170, 210), 'intensity': 0.35},
        'lip_gloss': 0.5,  # Extra glossy
    },
}

MAKEUP_KEYS = list(MAKEUP_LOOKS.keys())


def get_landmark_coords(face_landmarks, image_shape):
    """Extract pixel coordinates for all face landmarks."""
    h, w = image_shape[:2]
    coords = {}
    for idx, lm in enumerate(face_landmarks):
        x = int(lm.x * w)
        y = int(lm.y * h)
        coords[idx] = (x, y)
    return coords


def apply_makeup(image, face_landmarks, makeup_look):
    """
    Apply complete makeup look including all advanced effects.
    Order: Foundation -> Contour -> Color Makeup -> Blush -> Lip Gloss
    """
    result = image.copy()
    
    # Check if this is the "no makeup" look
    if not makeup_look.get('features') and not makeup_look.get('foundation'):
        return result
    
    # 1. Apply foundation (skin smoothing) first
    foundation_strength = makeup_look.get('foundation', 0)
    if foundation_strength > 0:
        warmth = makeup_look.get('foundation_warmth', 0)
        result = apply_foundation(result, face_landmarks, strength=foundation_strength, warmth=warmth)
    
    # 2. Apply contouring and highlighting
    if makeup_look.get('contour', False):
        result = apply_contour(result, face_landmarks)
    
    # 3. Apply color makeup (eyeshadow, eyeliner, lips, brows)
    features = makeup_look.get('features', {})
    if features:
        h, w = result.shape[:2]
        coords = get_landmark_coords(face_landmarks, result.shape)
        
        # Create makeup overlay
        overlay = result.copy()
        
        for region_name, style in features.items():
            if region_name not in FACE_REGIONS:
                continue
            
            region_indices = FACE_REGIONS[region_name]
            color = style['color']
            
            try:
                points = np.array([coords[idx] for idx in region_indices], dtype=np.int32)
                cv2.fillPoly(overlay, [points], color)
            except KeyError:
                continue
        
        # Blur for smooth edges
        overlay = cv2.GaussianBlur(overlay, (7, 7), 5)
        
        # Apply per-feature alpha
        for region_name, style in features.items():
            if region_name not in FACE_REGIONS:
                continue
            
            region_indices = FACE_REGIONS[region_name]
            alpha = style.get('alpha', 0.5)
            
            mask = np.zeros((h, w), dtype=np.uint8)
            try:
                points = np.array([coords[idx] for idx in region_indices], dtype=np.int32)
                cv2.fillPoly(mask, [points], 255)
            except KeyError:
                continue
            
            mask = cv2.GaussianBlur(mask, (15, 15), 10)
            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(float) / 255.0
            result = (result * (1 - mask_3ch * alpha) + overlay * mask_3ch * alpha).astype(np.uint8)
    
    # 4. Apply blush glow effect
    blush_config = makeup_look.get('blush_glow')
    if blush_config:
        color = blush_config.get('color', (180, 140, 200))
        intensity = blush_config.get('intensity', 0.25)
        result = apply_blush_glow(result, face_landmarks, color=color, intensity=intensity)
    
    # 5. Apply lip gloss last (shiny highlight on top)
    lip_gloss_intensity = makeup_look.get('lip_gloss', 0)
    if lip_gloss_intensity > 0:
        result = apply_lip_gloss(result, face_landmarks, intensity=lip_gloss_intensity)
    
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
                # OBJ faces are 1-indexed, handle formats like "f 1 2 3" or "f 1/1/1 2/2/2 3/3/3"
                face = []
                for p in parts[1:]:
                    idx = int(p.split('/')[0]) - 1
                    face.append(idx)
                faces.append(face)
    
    return np.array(vertices, dtype=np.float32), faces


# ============ FACE MESH BASED 3D TRANSFORM ============

def get_face_transform(face_landmarks, image_shape):
    """
    Extract 3D transformation directly from MediaPipe face mesh landmarks.
    Uses the mesh's native 3D coordinates for stable tracking.
    """
    h, w = image_shape[:2]
    
    # Key landmark indices
    FOREHEAD = 10      # Top of forehead
    CHIN = 152         # Bottom of chin
    LEFT_EYE = 263     # Left eye outer
    RIGHT_EYE = 33     # Right eye outer
    NOSE = 1           # Nose tip
    LEFT_CHEEK = 234   # Left side
    RIGHT_CHEEK = 454  # Right side
    
    # Get landmark positions (MediaPipe provides normalized x, y and relative z)
    def get_3d(idx):
        lm = face_landmarks[idx]
        return np.array([lm.x * w, lm.y * h, lm.z * w])
    
    forehead = get_3d(FOREHEAD)
    chin = get_3d(CHIN)
    left_eye = get_3d(LEFT_EYE)
    right_eye = get_3d(RIGHT_EYE)
    nose = get_3d(NOSE)
    left_cheek = get_3d(LEFT_CHEEK)
    right_cheek = get_3d(RIGHT_CHEEK)
    
    # Calculate face coordinate system axes
    # Up vector: from chin to forehead
    up = forehead - chin
    up = up / (np.linalg.norm(up) + 1e-6)
    
    # Right vector: from left to right eye
    right = right_eye - left_eye
    right = right / (np.linalg.norm(right) + 1e-6)
    
    # Forward vector: cross product (pointing out of face)
    forward = np.cross(right, up)
    forward = forward / (np.linalg.norm(forward) + 1e-6)
    
    # Recalculate up to ensure orthogonality
    up = np.cross(forward, right)
    up = up / (np.linalg.norm(up) + 1e-6)
    
    # Face scale (distance from forehead to chin)
    face_height = np.linalg.norm(forehead - chin)
    face_width = np.linalg.norm(right_eye - left_eye)
    
    # Head top position (above forehead along up vector)
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
    """
    Project 3D model using face mesh coordinate system.
    Model is placed at head top, oriented with face.
    
    model_offset: (right, up, forward) offset in face-relative coordinates
    """
    if len(vertices) == 0:
        return np.array([])
    
    origin = face_transform['origin']
    right = face_transform['right']
    up = face_transform['up']
    forward = face_transform['forward']
    face_height = face_transform['face_height']
    
    # Scale based on face size
    scale = (face_height / 120.0) * scale_factor
    
    # Apply model offset in face coordinate system
    offset_right, offset_up, offset_forward = model_offset
    offset_world = (right * offset_right + up * offset_up + forward * offset_forward) * scale
    adjusted_origin = origin + offset_world
    
    # Build rotation matrix from face axes
    # Model coords: X=right, Y=up (we flip), Z=forward
    rotation_matrix = np.column_stack([right, -up, forward])
    
    projected = []
    for v in vertices:
        # Scale vertex
        scaled = v * scale
        
        # Rotate to face orientation
        rotated = rotation_matrix @ scaled
        
        # Translate to head position (with offset applied)
        world_pos = adjusted_origin + rotated
        
        # Project to 2D (just take x, y since we're already in image coords)
        projected.append([int(world_pos[0]), int(world_pos[1])])
    
    return np.array(projected, dtype=np.int32)


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
    """Render 3D model with filled polygons (semi-transparent)."""
    if len(projected_points) == 0:
        return
    
    overlay = image.copy()
    
    # Sort faces by average Z (painter's algorithm - simple depth sorting)
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
    
    # Crown base ring
    n_points = 8
    radius = 30
    height = 0
    
    for i in range(n_points):
        angle = 2 * np.pi * i / n_points
        x = radius * np.cos(angle)
        z = radius * np.sin(angle)
        vertices.append([x, height, z])
    
    # Crown peaks
    peak_height = -40
    for i in range(n_points):
        angle = 2 * np.pi * (i + 0.5) / n_points
        x = radius * 0.9 * np.cos(angle)
        z = radius * 0.9 * np.sin(angle)
        vertices.append([x, peak_height, z])
    
    # Top ring (smaller)
    for i in range(n_points):
        angle = 2 * np.pi * i / n_points
        x = radius * 0.7 * np.cos(angle)
        z = radius * 0.7 * np.sin(angle)
        vertices.append([x, peak_height * 0.5, z])
    
    # Faces connecting base to peaks
    for i in range(n_points):
        next_i = (i + 1) % n_points
        # Triangle from base to peak
        faces.append([i, next_i, n_points + i])
        faces.append([next_i, n_points + next_i, n_points + i])
        # Connect peaks to top ring
        faces.append([n_points + i, n_points + next_i, 2*n_points + i])
        faces.append([n_points + next_i, 2*n_points + next_i, 2*n_points + i])
    
    return np.array(vertices, dtype=np.float32), faces


def create_hat_model():
    """Create a simple top hat 3D model."""
    vertices = []
    faces = []
    
    n_points = 16
    brim_radius = 45
    top_radius = 25
    hat_height = -50
    
    # Brim outer ring
    for i in range(n_points):
        angle = 2 * np.pi * i / n_points
        vertices.append([brim_radius * np.cos(angle), 0, brim_radius * np.sin(angle)])
    
    # Brim inner ring (where crown starts)
    for i in range(n_points):
        angle = 2 * np.pi * i / n_points
        vertices.append([top_radius * np.cos(angle), 0, top_radius * np.sin(angle)])
    
    # Crown top ring
    for i in range(n_points):
        angle = 2 * np.pi * i / n_points
        vertices.append([top_radius * np.cos(angle), hat_height, top_radius * np.sin(angle)])
    
    # Top center
    top_center_idx = len(vertices)
    vertices.append([0, hat_height, 0])
    
    # Brim faces
    for i in range(n_points):
        next_i = (i + 1) % n_points
        faces.append([i, next_i, n_points + next_i, n_points + i])
    
    # Crown side faces
    for i in range(n_points):
        next_i = (i + 1) % n_points
        faces.append([n_points + i, n_points + next_i, 2*n_points + next_i, 2*n_points + i])
    
    # Top faces
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
        n_segments = 6
        base_radius = 8
        
        for seg in range(n_segments + 1):
            t = seg / n_segments
            # Curved horn shape
            y = -t * 50
            x = x_offset + curve_dir * t * t * 20
            z = t * 10
            radius = base_radius * (1 - t * 0.8)
            
            # Create ring of vertices
            for i in range(6):
                angle = 2 * np.pi * i / 6
                vx = x + radius * np.cos(angle)
                vz = z + radius * np.sin(angle)
                vertices.append([vx, y, vz])
        
        # Create faces between rings
        for seg in range(n_segments):
            for i in range(6):
                next_i = (i + 1) % 6
                idx = base_idx + seg * 6
                faces.append([
                    idx + i, idx + next_i, 
                    idx + 6 + next_i, idx + 6 + i
                ])
    
    # Left and right horns
    add_horn(-30, -1)
    add_horn(30, 1)
    
    return np.array(vertices, dtype=np.float32), faces


def create_halo_model():
    """Create a halo/angel ring 3D model."""
    vertices = []
    faces = []
    
    n_points = 24
    outer_radius = 40
    inner_radius = 32
    ring_height = -50  # Above head
    thickness = 3
    
    # Outer ring top
    for i in range(n_points):
        angle = 2 * np.pi * i / n_points
        vertices.append([outer_radius * np.cos(angle), ring_height - thickness, outer_radius * np.sin(angle)])
    
    # Inner ring top
    for i in range(n_points):
        angle = 2 * np.pi * i / n_points
        vertices.append([inner_radius * np.cos(angle), ring_height - thickness, inner_radius * np.sin(angle)])
    
    # Outer ring bottom
    for i in range(n_points):
        angle = 2 * np.pi * i / n_points
        vertices.append([outer_radius * np.cos(angle), ring_height + thickness, outer_radius * np.sin(angle)])
    
    # Inner ring bottom  
    for i in range(n_points):
        angle = 2 * np.pi * i / n_points
        vertices.append([inner_radius * np.cos(angle), ring_height + thickness, inner_radius * np.sin(angle)])
    
    # Top face
    for i in range(n_points):
        next_i = (i + 1) % n_points
        faces.append([i, next_i, n_points + next_i, n_points + i])
    
    # Bottom face
    for i in range(n_points):
        next_i = (i + 1) % n_points
        faces.append([2*n_points + i, 3*n_points + i, 3*n_points + next_i, 2*n_points + next_i])
    
    # Outer edge
    for i in range(n_points):
        next_i = (i + 1) % n_points
        faces.append([i, 2*n_points + i, 2*n_points + next_i, next_i])
    
    # Inner edge
    for i in range(n_points):
        next_i = (i + 1) % n_points
        faces.append([n_points + i, n_points + next_i, 3*n_points + next_i, 3*n_points + i])
    
    return np.array(vertices, dtype=np.float32), faces


def create_party_hat_model():
    """Create a party/cone hat 3D model."""
    vertices = []
    faces = []
    
    n_points = 12
    base_radius = 30
    hat_height = -70
    
    # Base ring
    for i in range(n_points):
        angle = 2 * np.pi * i / n_points
        vertices.append([base_radius * np.cos(angle), 0, base_radius * np.sin(angle)])
    
    # Cone tip
    tip_idx = len(vertices)
    vertices.append([0, hat_height, 0])
    
    # Cone faces
    for i in range(n_points):
        next_i = (i + 1) % n_points
        faces.append([i, next_i, tip_idx])
    
    # Base (optional, for closed cone)
    base_center_idx = len(vertices)
    vertices.append([0, 0, 0])
    for i in range(n_points):
        next_i = (i + 1) % n_points
        faces.append([next_i, i, base_center_idx])
    
    return np.array(vertices, dtype=np.float32), faces


# ============ MAIN APPLICATION ============

# Available models with their colors and position offsets
# Offset format: (right, up, forward) - relative to head top position
# Positive right = model shifts right, positive up = higher, positive forward = towards camera
MODELS = {
    'crown': {
        'create': create_crown_model,
        'color': (0, 215, 255),
        'name': 'Golden Crown',
        'offset': (0, 5, 60),  # Sits on top of head
    },
    'hat': {
        'create': create_hat_model,
        'color': (40, 40, 40),
        'name': 'Top Hat',
        'offset': (0, 15, 60),  # Tall hat, slightly back
    },
    'horns': {
        'create': create_horns_model,
        'color': (0, 0, 180),
        'name': 'Devil Horns',
        'offset': (0, -5, 30),  # Lower on forehead
    },
    'halo': {
        'create': create_halo_model,
        'color': (0, 255, 255),
        'name': 'Angel Halo',
        'offset': (0, 20, 60),  # Floating above head
    },
    'party': {
        'create': create_party_hat_model,
        'color': (255, 0, 128),
        'name': 'Party Hat',
        'offset': (0, 5, 60),  # Slightly tilted forward
    },
    'wings': {
        'create': lambda: load_obj('models/wings.obj'),
        'color': (255, 255, 255),
        'name': 'Wings',
        'offset': (0, 150, 80),  # Slightly tilted forward
    },
}

# List for cycling through models
MODEL_KEYS = list(MODELS.keys())

def draw_face_mesh_landmarks(image, face_landmarks):
    """Draw face mesh landmarks on the image (simplified version without solutions API)."""
    h, w = image.shape[:2]
    # Draw landmarks as small circles
    for lm in face_landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        cv2.circle(image, (x, y), 1, (0, 255, 0), -1)


def main():
    # Ensure model file exists
    ensure_model_exists()
    
    # Current model index
    current_model_idx = 0
    show_face_mesh = False
    render_filled = True
    model_scale = 1.0
    
    # Current makeup index
    current_makeup_idx = 0
    
    # Load initial model
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
    
    # For webcam input:
    cap = cv2.VideoCapture(0)
    
    # Set camera to highest resolution (request 4K, camera will use max supported)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # Use MJPG for higher resolutions
    
    # Get actual camera resolution (may be lower than requested if not supported)
    cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cam_fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    
    print(f"Camera resolution: {cam_width}x{cam_height} @ {cam_fps}fps")
    
    print("\n=== 3D Head Model AR Demo ===")
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
    
    # Virtual camera setup
    vcam = None
    vcam_enabled = False
    if VIRTUALCAM_AVAILABLE:
        try:
            vcam = pyvirtualcam.Camera(width=cam_width, height=cam_height, fps=cam_fps, fmt=pyvirtualcam.PixelFormat.BGR)
            vcam_enabled = True
            print(f"Virtual camera started: {vcam.device}")
            print(f"Resolution: {cam_width}x{cam_height} @ {cam_fps}fps")
        except Exception as e:
            print(f"Could not start virtual camera: {e}")
            print("Make sure v4l2loopback is loaded: sudo modprobe v4l2loopback devices=1")
            vcam = None
    
    # Create FaceLandmarker using Tasks API
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False
    )
    
    face_landmarker = vision.FaceLandmarker.create_from_options(options)
    
    try:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
            
            # Flip for selfie view
            image = cv2.flip(image, 1)
            
            # Process face mesh using Tasks API
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            results = face_landmarker.detect(mp_image)
            
            if results.face_landmarks:
                for face_landmarks in results.face_landmarks:
                    # Apply makeup first (before 3D model)
                    image = apply_makeup(image, face_landmarks, current_makeup)
                    
                    # Optionally draw face mesh
                    if show_face_mesh:
                        draw_face_mesh_landmarks(image, face_landmarks)
                    
                    # Get face transform directly from mesh landmarks
                    face_transform = get_face_transform(face_landmarks, image.shape)
                    
                    # Project 3D model using face coordinate system
                    projected = project_model_from_face(vertices, face_transform, model_scale, model_offset)
                    
                    # Render the model
                    if len(projected) > 0:
                        if render_filled:
                            render_model_filled(image, projected, faces, model_color, alpha=0.7)
                        render_model_wireframe(image, projected, faces, model_color, thickness=2)
            
            # Send clean frame to virtual camera (no UI overlay)
            if vcam is not None and vcam_enabled:
                vcam.send(image)
                vcam.sleep_until_next_frame()
            
            # Create preview with UI overlay
            preview = image.copy()
            cv2.rectangle(preview, (10, 10), (320, 115), (0, 0, 0), -1)
            cv2.rectangle(preview, (10, 10), (320, 115), model_color, 2)
            cv2.putText(preview, f"Hat: {model_name}", (20, 32), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(preview, f"Makeup: {current_makeup['name']}", (20, 52), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 255), 1)
            cv2.putText(preview, f"Scale: {model_scale:.1f}x", (20, 72), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            vcam_status = "ON" if (vcam is not None and vcam_enabled) else "OFF"
            cv2.putText(preview, f"[N/P] Hat | [</>] Makeup | [V]Cam:{vcam_status}", (20, 95), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
            cv2.putText(preview, f"[F] Fill | [M] Mesh | [+/-] Scale", (20, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
            
            cv2.imshow('3D Head Model AR', preview)
            
            # Handle keyboard input
            key = cv2.waitKey(5) & 0xFF
            
            if key == 27:  # ESC
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
        # Cleanup
        face_landmarker.close()
        cap.release()
        if vcam is not None:
            vcam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
