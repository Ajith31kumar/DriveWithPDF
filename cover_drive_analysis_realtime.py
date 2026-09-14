import os
import io
import json
import time
import math
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors

import mediapipe as mp
from ultralytics import YOLO

from models.custom_detector_v2 import CustomCricketDetectorV2


# ============================================================
# PATHS / I/O
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = str(PROJECT_DIR / "output")
CONFIG_DIR = str(PROJECT_DIR / "config")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)

ANNOTATED_VIDEO_PATH = os.path.join(
    OUTPUT_DIR, "annotated_bonus.mp4"
)

FINAL_GRAPH_PNG_PATH = os.path.join(
    OUTPUT_DIR, "smoothness_final.png"
)

EVAL_JSON_PATH = os.path.join(
    OUTPUT_DIR, "evaluation.json"
)

PER_FRAME_CSV_PATH = os.path.join(
    OUTPUT_DIR, "per_frame_metrics.csv"
)

PDF_REPORT_PATH = os.path.join(
    OUTPUT_DIR, "report.pdf"
)

TARGETS_JSON_PATH = os.path.join(
    CONFIG_DIR, "targets.json"
)

# Current MediaPipe Tasks pose model.
# MediaPipe's current Python API uses PoseLandmarker rather than
# the legacy mp.solutions.pose API.
POSE_MODEL_PATH = os.path.join(
    CONFIG_DIR,
    "pose_landmarker_full.task"
)

# ============================================================
# FEATURE FLAGS
# ============================================================
# Legacy features stay in the codebase for backward compatibility.
# The new clean-analysis graph is the default visible result.
SHOW_LEGACY_GRAPH = False
SHOW_NEW_GRAPH = True
NEW_GRAPH_PNG_PATH = os.path.join(OUTPUT_DIR, "movement_analysis_v2.png")
LEGACY_GRAPH_PNG_PATH = os.path.join(OUTPUT_DIR, "smoothness_legacy.png")


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/"
    "float16/latest/pose_landmarker_full.task"
)



# ============================================================
# BALL / BAT MODEL
# ============================================================

# Custom V2 Bat + Ball detector trained in TestBatCustom.
CUSTOM_MODEL_PATH = (
    r"C:\Users\Admin\Desktop\cusotm\TestBatCustom\runs\custom_v2\custom_detector_v2_best.pt"
)

BAT_CLASS_ID = 0
BALL_CLASS_ID = 1

# Confidence used by the custom PyTorch V2 detector.
BALL_BAT_CONFIDENCE = 0.10

# V2 model input size.
CUSTOM_IMAGE_SIZE = 640

CUSTOM_DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Compatibility names kept for the existing backend.
BALL_IMGSZ = CUSTOM_IMAGE_SIZE
BAT_IMGSZ = CUSTOM_IMAGE_SIZE

# ROI padding around the detected batsman.
BAT_ROI_PADDING_X = 180
BAT_ROI_PADDING_Y = 220

# Ball search uses a larger region around the batsman plus
# a little extra space in the likely ball approach area.
BALL_ROI_PADDING_X = 420
BALL_ROI_PADDING_Y = 320

# Draw/track the best candidate in the current frame.
ENABLE_BALL_BAT_TRACKING = True

# ============================================================
# DEFAULT TARGETS
# ============================================================

DEFAULT_TARGETS = {
    "elbow_angle_deg": [120, 170],
    "spine_lean_deg": [0, 20],
    "head_knee_diff_px": [0, 40],
    "foot_direction_deg": [-45, 45],
    "bat_angle_deg_at_impact": [-20, 20]
}


def ensure_targets_config(path=TARGETS_JSON_PATH):
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_TARGETS, f, indent=4)

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# MEDIA PIPE TASK MODEL
# ============================================================

def ensure_pose_model():
    """
    Download the MediaPipe PoseLandmarker task model once.
    """
    if os.path.exists(POSE_MODEL_PATH):
        return POSE_MODEL_PATH

    print("MediaPipe pose model not found.")
    print("Downloading pose_landmarker_full.task ...")

    try:
        urllib.request.urlretrieve(
            MODEL_URL,
            POSE_MODEL_PATH
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not download MediaPipe pose model.\n"
            f"URL: {MODEL_URL}\n"
            f"Error: {exc}"
        ) from exc

    if not os.path.exists(POSE_MODEL_PATH):
        raise RuntimeError(
            "Pose model download completed without creating the file:\n"
            f"{POSE_MODEL_PATH}"
        )

    return POSE_MODEL_PATH


def create_pose_landmarker():
    """
    Create the current MediaPipe PoseLandmarker in VIDEO mode.

    VIDEO mode is important here because we are processing a sequence
    of video frames with monotonically increasing timestamps.
    """
    model_path = ensure_pose_model()

    base_options = mp.tasks.BaseOptions(
        model_asset_path=model_path
    )

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False
    )

    return mp.tasks.vision.PoseLandmarker.create_from_options(
        options
    )


# ============================================================
# POSE LANDMARK INDEXES
# MediaPipe pose landmark indexing
# ============================================================

POSE_IDS = {
    "nose": 0,
    "right_shoulder": 12,
    "right_elbow": 14,
    "right_wrist": 16,
    "left_wrist": 15,
    "right_hip": 24,
    "right_knee": 26,
    "right_ankle": 28,
}


# A practical subset of the standard body connections.
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),

    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),

    (11, 23), (12, 24),
    (23, 24),

    (23, 25), (25, 27),
    (24, 26), (26, 28),

    (27, 29), (29, 31),
    (28, 30), (30, 32)
]


def normalized_landmark_to_px(lm, width, height):
    x = int(np.clip(lm.x, 0.0, 1.0) * width)
    y = int(np.clip(lm.y, 0.0, 1.0) * height)
    return x, y


def draw_pose_landmarks(frame, landmarks):
    """
    Draw pose landmarks without using the removed legacy
    mp.solutions.drawing_utils API.
    """
    height, width = frame.shape[:2]

    points = {}

    for idx, lm in enumerate(landmarks):
        points[idx] = normalized_landmark_to_px(
            lm, width, height
        )

    # Connections
    for a, b in POSE_CONNECTIONS:
        if a in points and b in points:
            cv2.line(
                frame,
                points[a],
                points[b],
                (0, 255, 0),
                2
            )

    # Points
    for idx, point in points.items():
        cv2.circle(
            frame,
            point,
            3,
            (0, 180, 255),
            -1
        )

    return frame


def get_joints_from_landmarks(landmarks, width, height):
    joints = {}

    for name, idx in POSE_IDS.items():
        if idx >= len(landmarks):
            continue

        lm = landmarks[idx]

        # Ignore very low-confidence / invalid points where possible.
        visibility = getattr(lm, "visibility", 1.0)

        if visibility < 0.25:
            continue

        joints[name] = normalized_landmark_to_px(
            lm,
            width,
            height
        )

    return joints


# ============================================================
# METRIC HELPERS
# ============================================================

def angle_ABC(a, b, c):
    a = np.array(a, float)
    b = np.array(b, float)
    c = np.array(c, float)

    if (
        np.linalg.norm(a - b) < 1e-6
        or
        np.linalg.norm(c - b) < 1e-6
    ):
        return None

    ba = a - b
    bc = c - b

    cosang = (
        np.dot(ba, bc)
        /
        (
            np.linalg.norm(ba)
            *
            np.linalg.norm(bc)
        )
    )

    cosang = np.clip(
        cosang,
        -1.0,
        1.0
    )

    return float(
        np.degrees(
            np.arccos(cosang)
        )
    )


def spine_lean_deg(hip, shoulder):
    dx = shoulder[0] - hip[0]
    dy = shoulder[1] - hip[1]

    angle = float(
        np.degrees(
            np.arctan2(dx, dy)
        )
    )

    angle = abs(angle)

    if angle > 90:
        angle = 180 - angle

    return round(angle, 1)


def head_knee_diff_px(nose, knee):
    return float(
        abs(nose[0] - knee[0])
    )


def foot_direction_deg(ankle, knee):
    dx = knee[0] - ankle[0]
    dy = knee[1] - ankle[1]

    return float(
        np.degrees(
            np.arctan2(dy, dx)
        )
    )


def normalize_angle_deg(angle):
    if angle is None:
        return None

    a = ((angle + 180) % 360) - 180
    return round(a, 1)


def normalized_head_knee(nose, knee, hip):
    torso_h = math.hypot(
        nose[0] - hip[0],
        nose[1] - hip[1]
    )

    if torso_h < 1e-6:
        return None

    raw_px = abs(
        nose[0] - knee[0]
    )

    return round(
        (raw_px / torso_h) * 100.0,
        1
    )



# ============================================================
# BALL / BAT CUSTOM V2 DETECTION + ROI TRACKING
# ============================================================

_BALL_BAT_MODEL = None


def load_ball_bat_model():
    """
    Load the trained Custom V2 Bat + Ball detector once.

    The checkpoint is produced by train_custom_v2.py and contains
    a ``model_state_dict`` entry.
    """
    global _BALL_BAT_MODEL

    if _BALL_BAT_MODEL is not None:
        return _BALL_BAT_MODEL

    if not os.path.isfile(CUSTOM_MODEL_PATH):
        raise FileNotFoundError(
            "Custom V2 Bat/Ball model not found:\n"
            f"{CUSTOM_MODEL_PATH}"
        )

    checkpoint = torch.load(
        CUSTOM_MODEL_PATH,
        map_location=CUSTOM_DEVICE,
    )

    model = CustomCricketDetectorV2(
        num_classes=2,
        num_slots=2,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(checkpoint)

    model.to(CUSTOM_DEVICE)
    model.eval()

    _BALL_BAT_MODEL = model

    return _BALL_BAT_MODEL


def _custom_preprocess(frame):
    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB,
    )

    resized = cv2.resize(
        rgb,
        (CUSTOM_IMAGE_SIZE, CUSTOM_IMAGE_SIZE),
    )

    tensor = (
        torch.from_numpy(resized)
        .permute(2, 0, 1)
        .float()
        / 255.0
    )

    return tensor.unsqueeze(0).to(CUSTOM_DEVICE)


def _custom_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    intersection = iw * ih

    area_a = (
        max(0.0, ax2 - ax1)
        * max(0.0, ay2 - ay1)
    )

    area_b = (
        max(0.0, bx2 - bx1)
        * max(0.0, by2 - by1)
    )

    union = area_a + area_b - intersection

    if union <= 0.0:
        return 0.0

    return intersection / union


def _custom_nms(
    detections,
    iou_threshold=0.45,
):
    detections = sorted(
        detections,
        key=lambda item: item["confidence"],
        reverse=True,
    )

    selected = []

    for detection in detections:
        keep = True

        for existing in selected:
            if (
                detection["class_id"]
                != existing["class_id"]
            ):
                continue

            if (
                _custom_iou(
                    detection["box"],
                    existing["box"],
                )
                > iou_threshold
            ):
                keep = False
                break

        if keep:
            selected.append(detection)

    return selected


def _custom_decode_predictions(raw_output):
    """
    Decode Custom V2 output.

    V2 output shape:
        [B, 14, 80, 80]

    Per grid cell / slot:
        tx, ty, tw, th, objectness, class0, class1

    Returns:
        boxes         [B, H, W, S, 4] normalized xyxy
        objectness    [B, H, W, S]
        class_probs   [B, H, W, S, 2]
    """
    batch_size, channels, grid_h, grid_w = raw_output.shape

    num_slots = 2
    num_classes = 2
    values_per_slot = 5 + num_classes

    expected_channels = (
        num_slots * values_per_slot
    )

    if channels != expected_channels:
        raise RuntimeError(
            "Unexpected Custom V2 output shape: "
            f"{tuple(raw_output.shape)}. "
            f"Expected {expected_channels} channels."
        )

    prediction = raw_output.view(
        batch_size,
        num_slots,
        values_per_slot,
        grid_h,
        grid_w,
    )

    tx = torch.sigmoid(
        prediction[:, :, 0]
    )
    ty = torch.sigmoid(
        prediction[:, :, 1]
    )

    tw = torch.clamp(
        prediction[:, :, 2],
        -6.0,
        6.0,
    )
    th = torch.clamp(
        prediction[:, :, 3],
        -6.0,
        6.0,
    )

    objectness = torch.sigmoid(
        prediction[:, :, 4]
    )

    class_probs = torch.softmax(
        prediction[:, :, 5:],
        dim=2,
    )

    grid_y, grid_x = torch.meshgrid(
        torch.arange(
            grid_h,
            device=raw_output.device,
            dtype=raw_output.dtype,
        ),
        torch.arange(
            grid_w,
            device=raw_output.device,
            dtype=raw_output.dtype,
        ),
        indexing="ij",
    )

    grid_x = grid_x.view(
        1, 1, grid_h, grid_w
    )
    grid_y = grid_y.view(
        1, 1, grid_h, grid_w
    )

    center_x = (
        tx + grid_x
    ) / float(grid_w)

    center_y = (
        ty + grid_y
    ) / float(grid_h)

    box_w = (
        torch.exp(tw)
        / float(grid_w)
    )

    box_h = (
        torch.exp(th)
        / float(grid_h)
    )

    x1 = center_x - (box_w / 2.0)
    y1 = center_y - (box_h / 2.0)
    x2 = center_x + (box_w / 2.0)
    y2 = center_y + (box_h / 2.0)

    boxes = torch.stack(
        [x1, y1, x2, y2],
        dim=-1,
    )

    boxes = boxes.permute(
        0, 2, 3, 1, 4
    )

    objectness = objectness.permute(
        0, 2, 3, 1
    )

    class_probs = class_probs.permute(
        0, 3, 4, 1, 2
    )

    return (
        boxes,
        objectness,
        class_probs,
    )


def _custom_predict_detections(
    frame,
    model,
):
    """Run Custom V2 and return the backend detection dictionary shape."""
    frame_height, frame_width = frame.shape[:2]

    tensor = _custom_preprocess(frame)

    with torch.no_grad():
        raw_output = model(tensor)

    boxes, objectness, class_probs = (
        _custom_decode_predictions(raw_output)
    )

    boxes = boxes[0].detach().cpu().numpy()
    objectness = objectness[0].detach().cpu().numpy()
    class_probs = class_probs[0].detach().cpu().numpy()

    detections = []

    grid_h, grid_w, num_slots = objectness.shape

    for gy in range(grid_h):
        for gx in range(grid_w):
            for slot in range(num_slots):

                class_id = int(
                    np.argmax(
                        class_probs[gy, gx, slot]
                    )
                )

                class_conf = float(
                    class_probs[
                        gy,
                        gx,
                        slot,
                        class_id,
                    ]
                )

                confidence = float(
                    objectness[
                        gy,
                        gx,
                        slot,
                    ]
                    * class_conf
                )

                if confidence < BALL_BAT_CONFIDENCE:
                    continue

                x1n, y1n, x2n, y2n = (
                    boxes[
                        gy,
                        gx,
                        slot,
                    ]
                )

                x1n = float(
                    np.clip(x1n, 0.0, 1.0)
                )
                y1n = float(
                    np.clip(y1n, 0.0, 1.0)
                )
                x2n = float(
                    np.clip(x2n, 0.0, 1.0)
                )
                y2n = float(
                    np.clip(y2n, 0.0, 1.0)
                )

                x1 = x1n * frame_width
                y1 = y1n * frame_height
                x2 = x2n * frame_width
                y2 = y2n * frame_height

                if x2 <= x1 or y2 <= y1:
                    continue

                detections.append(
                    {
                        "class_id": class_id,
                        "confidence": confidence,
                        "box": (
                            x1,
                            y1,
                            x2,
                            y2,
                        ),
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2),
                        "center_x": (
                            x1 + x2
                        ) / 2.0,
                        "center_y": (
                            y1 + y2
                        ) / 2.0,
                        "width": max(
                            0,
                            int(x2 - x1),
                        ),
                        "height": max(
                            0,
                            int(y2 - y1),
                        ),
                    }
                )

    return _custom_nms(
        detections,
        iou_threshold=0.45,
    )


def select_best_detection(
    detections,
    class_id,
):
    """
    Select the highest-confidence Custom V2 detection
    for a requested class.
    """
    best = None
    best_conf = 0.0

    if not detections:
        return None

    for detection in detections:
        if detection["class_id"] != class_id:
            continue

        conf = float(
            detection["confidence"]
        )

        if conf < BALL_BAT_CONFIDENCE:
            continue

        if conf > best_conf:
            best_conf = conf
            best = dict(detection)

    return best


def clamp_box(x1, y1, x2, y2, width, height):
    return (
        max(0, int(x1)),
        max(0, int(y1)),
        min(width, int(x2)),
        min(height, int(y2)),
    )


def batsman_roi_from_joints(
    joints,
    width,
    height
):
    """
    Build a generous ROI around the detected batsman.

    The ROI is used mainly to improve bat detection by making the
    small bat larger relative to the detector input.
    """
    points = [
        joints.get("nose"),
        joints.get("right_shoulder"),
        joints.get("right_hip"),
        joints.get("right_knee"),
        joints.get("right_ankle"),
        joints.get("right_wrist"),
        joints.get("left_wrist"),
    ]

    points = [
        p for p in points
        if p is not None
    ]

    if len(points) < 3:
        return None

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    x1 = min(xs) - BAT_ROI_PADDING_X
    y1 = min(ys) - BAT_ROI_PADDING_Y
    x2 = max(xs) + BAT_ROI_PADDING_X
    y2 = max(ys) + BAT_ROI_PADDING_Y

    return clamp_box(
        x1, y1, x2, y2,
        width, height
    )


def ball_search_roi(
    joints,
    width,
    height
):
    """
    Larger ROI for ball detection. The ball can be outside the body
    box, so this ROI is deliberately wider than the bat ROI.
    """
    body_roi = batsman_roi_from_joints(
        joints,
        width,
        height
    )

    if body_roi is None:
        return None

    x1, y1, x2, y2 = body_roi

    x1 -= BALL_ROI_PADDING_X
    y1 -= BALL_ROI_PADDING_Y
    x2 += BALL_ROI_PADDING_X
    y2 += BALL_ROI_PADDING_Y

    return clamp_box(
        x1, y1, x2, y2,
        width, height
    )


def detect_bat_in_batsman_roi(
    frame,
    joints,
    model
):
    """Detect BAT using the Custom V2 detector inside the batsman ROI."""
    height, width = frame.shape[:2]

    roi = batsman_roi_from_joints(
        joints,
        width,
        height
    )

    if roi is None:
        return None, None

    x1, y1, x2, y2 = roi

    crop = frame[
        y1:y2,
        x1:x2
    ]

    if crop.size == 0:
        return None, roi

    detections = _custom_predict_detections(
        crop,
        model,
    )

    detection = select_best_detection(
        detections,
        BAT_CLASS_ID,
    )

    if detection is None:
        return None, roi

    detection["x1"] += x1
    detection["x2"] += x1
    detection["y1"] += y1
    detection["y2"] += y1

    detection["center_x"] += x1
    detection["center_y"] += y1

    detection["box"] = (
        detection["x1"],
        detection["y1"],
        detection["x2"],
        detection["y2"],
    )

    return detection, roi


def detect_ball_in_broad_roi(
    frame,
    joints,
    model
):
    """Detect BALL using the Custom V2 detector in a broad ROI."""
    height, width = frame.shape[:2]

    roi = ball_search_roi(
        joints,
        width,
        height
    )

    if roi is None:
        crop = frame
        x1 = y1 = 0
        roi = (0, 0, width, height)
    else:
        x1, y1, x2, y2 = roi

        crop = frame[
            y1:y2,
            x1:x2
        ]

        if crop.size == 0:
            return None, roi

    detections = _custom_predict_detections(
        crop,
        model,
    )

    detection = select_best_detection(
        detections,
        BALL_CLASS_ID,
    )

    if detection is None:
        return None, roi

    detection["x1"] += x1
    detection["x2"] += x1
    detection["y1"] += y1
    detection["y2"] += y1

    detection["center_x"] += x1
    detection["center_y"] += y1

    detection["box"] = (
        detection["x1"],
        detection["y1"],
        detection["x2"],
        detection["y2"],
    )

    return detection, roi


def ball_bat_distance(
    ball_detection,
    bat_detection
):
    if not ball_detection or not bat_detection:
        return None

    return float(
        math.hypot(
            ball_detection["center_x"]
            - bat_detection["center_x"],

            ball_detection["center_y"]
            - bat_detection["center_y"],
        )
    )


def draw_detection(
    frame,
    detection,
    label,
    color
):
    if detection is None:
        return

    x1 = detection["x1"]
    y1 = detection["y1"]
    x2 = detection["x2"]
    y2 = detection["y2"]

    conf = detection["confidence"]

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        2
    )

    cv2.circle(
        frame,
        (
            int(detection["center_x"]),
            int(detection["center_y"])
        ),
        4,
        color,
        -1
    )

    cv2.putText(
        frame,
        f"{label} {conf:.2f}",
        (
            max(8, x1),
            max(22, y1 - 8)
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2
    )


# ============================================================
# BAT DETECTION / TRACKING
# ============================================================

def estimate_bat_line(
    frame_bgr,
    joints,
    box=120
):
    """
    ROI around midpoint between wrists -> Canny -> HoughLinesP.
    Returns:
        (angle_deg, (xA, yA, xB, yB))
    or:
        (None, None)
    """

    if (
        "right_wrist" not in joints
        or
        "left_wrist" not in joints
    ):
        return None, None

    (x1, y1) = joints["right_wrist"]
    (x2, y2) = joints["left_wrist"]

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    height, width = frame_bgr.shape[:2]

    x0 = max(0, cx - box)
    y0 = max(0, cy - box)

    x_end = min(width, cx + box)
    y_end = min(height, cy + box)

    roi = frame_bgr[
        y0:y_end,
        x0:x_end
    ]

    if roi.size == 0:
        return None, None

    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )

    edges = cv2.Canny(
        gray,
        60,
        180
    )

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=30,
        minLineLength=30,
        maxLineGap=10
    )

    if lines is None:
        return None, None

    best = None
    best_len = 0

    for line in lines:

        xa, ya, xb, yb = line[0]

        length = (
            (xb - xa) ** 2
            +
            (yb - ya) ** 2
        )

        if length > best_len:
            best_len = length
            best = (
                xa,
                ya,
                xb,
                yb
            )

    if best is None:
        return None, None

    xa, ya, xb, yb = best

    xa += x0
    ya += y0
    xb += x0
    yb += y0

    angle = float(
        np.degrees(
            np.arctan2(
                yb - ya,
                xb - xa
            )
        )
    )

    return angle, (
        xa,
        ya,
        xb,
        yb
    )


def path_straightness(angles_deg):
    arr = np.array(
        [
            a
            for a in angles_deg
            if a is not None
        ],
        float
    )

    if arr.size < 5:
        return None

    variance = float(
        np.var(arr)
    )

    return float(
        np.clip(
            1.0 / (1.0 + variance / 400.0),
            0.0,
            1.0
        )
    )


# ============================================================
# SMOOTHNESS
# ============================================================

def compute_smoothness(series):
    vals = np.array(
        [
            np.nan if v is None else float(v)
            for v in series
        ],
        float
    )

    for i in range(1, len(vals)):

        if np.isnan(vals[i]):
            vals[i] = vals[i - 1]

    if (
        len(vals)
        and
        np.isnan(vals[0])
    ):
        vals[0] = 0.0

    deltas = (
        np.abs(np.diff(vals))
        if len(vals) > 1
        else np.array([])
    )

    variance = (
        float(np.nanvar(vals))
        if len(vals)
        else 0.0
    )

    return (
        vals,
        deltas,
        variance
    )



# ============================================================
# PHASE 1: RELIABILITY + CLEAN GRAPH (NEW)
# ============================================================

def compute_clean_series(series):
    """
    Keep missing pose values as NaN.
    Unlike the legacy graph, this function does NOT carry the
    previous value forward. This prevents artificial flat lines.
    """
    return np.array(
        [
            np.nan if value is None else float(value)
            for value in series
        ],
        dtype=float,
    )


def compute_validity(series):
    """
    Return valid count, total count and coverage percentage.
    """
    total = len(series)
    valid = sum(
        1
        for value in series
        if value is not None
    )

    coverage = (
        (valid / total) * 100.0
        if total
        else 0.0
    )

    return valid, total, round(coverage, 1)


def normalize_phase_markers(phases):
    """
    Convert the current phase frame indexes into safe,
    chronologically ordered markers for visualization.
    """
    if not phases:
        return {}

    keys = [
        "stance_start",
        "stride_start",
        "downswing_start",
        "impact_frame",
        "follow_through_end",
        "recovery_start",
        "last_frame",
    ]

    result = {}
    previous = 0

    for key in keys:
        value = phases.get(key)

        if value is None:
            continue

        try:
            value = int(value)
        except (TypeError, ValueError):
            continue

        value = max(previous, value)
        result[key] = value
        previous = value

    return result


def plot_clean_movement_analysis(
    time_list,
    elbow_series,
    spine_series,
    wrist_speed_series,
    phases,
    fps,
    output_path,
):
    """
    New graph:
      - Elbow angle
      - Spine lean
      - Wrist speed (normalized to percentage scale)
      - Phase markers
      - No forward-filling of missing pose values
    """
    if not time_list:
        return

    t = np.asarray(time_list, dtype=float)
    elbow = compute_clean_series(elbow_series)
    spine = compute_clean_series(spine_series)

    wrist = np.asarray(
        [
            np.nan if v is None else float(v)
            for v in wrist_speed_series
        ],
        dtype=float,
    )

    # Normalize wrist speed only for visual overlay so it is
    # comparable to the angle axis.
    finite_wrist = wrist[np.isfinite(wrist)]
    if finite_wrist.size and np.nanmax(finite_wrist) > 0:
        wrist_plot = (
            wrist / np.nanmax(finite_wrist)
        ) * 180.0
    else:
        wrist_plot = np.full_like(
            wrist,
            np.nan,
            dtype=float,
        )

    fig, ax1 = plt.subplots(
        figsize=(14, 6)
    )

    ax1.plot(
        t,
        elbow,
        label="Elbow angle",
        linewidth=1.8,
    )

    ax1.plot(
        t,
        spine,
        label="Spine lean",
        linewidth=1.8,
    )

    ax1.plot(
        t,
        wrist_plot,
        label="Wrist speed (normalized)",
        linewidth=1.5,
        linestyle="--",
    )

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Angle / normalized wrist-speed scale")
    ax1.set_title(
        "VyronAI Movement Analysis – Clean Temporal View"
    )
    ax1.grid(
        True,
        alpha=0.2,
    )

    # Phase markers
    phase_markers = normalize_phase_markers(phases)

    labels = {
        "stance_start": "Stance",
        "stride_start": "Stride",
        "downswing_start": "Downswing",
        "impact_frame": "Impact",
        "follow_through_end": "Follow-through",
        "recovery_start": "Recovery",
    }

    for key, frame_no in phase_markers.items():

        if key == "last_frame":
            continue

        if fps and fps > 0:
            x = frame_no / fps
        else:
            x = None

        if x is None:
            continue

        ax1.axvline(
            x,
            linestyle=":",
            linewidth=1.2,
            alpha=0.6,
        )

        ax1.text(
            x,
            0.98,
            labels.get(key, key),
            transform=ax1.get_xaxis_transform(),
            rotation=90,
            va="top",
            ha="right",
            fontsize=8,
        )

    ax1.legend(
        loc="upper right"
    )

    fig.tight_layout()

    # Remove any stale/corrupt file before saving.
    try:
        if os.path.exists(output_path):
            os.remove(output_path)
    except OSError:
        pass

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
        format="png",
    )

    plt.close(fig)

    # Verify that a real, non-empty PNG was created.
    if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError(
            f"Movement graph was not written correctly: {output_path}"
        )


def clean_phase_order(phases, n_frames):
    """
    Preserve the legacy phase detector, but make the output
    chronology safe for the new analysis view.
    """
    if not phases:
        return {}

    def safe_int(value, fallback):
        try:
            return int(value)
        except (TypeError, ValueError):
            return fallback

    ordered_keys = [
        "stance_start",
        "stride_start",
        "downswing_start",
        "impact_frame",
        "follow_through_end",
        "recovery_start",
        "last_frame",
    ]

    result = {}
    previous = 0

    for key in ordered_keys:

        raw = phases.get(key)

        if key == "last_frame":
            value = (
                n_frames - 1
                if raw is None
                else safe_int(raw, n_frames - 1)
            )
        else:
            value = safe_int(
                raw,
                previous,
            )

        value = max(
            previous,
            min(value, max(0, n_frames - 1)),
        )

        result[key] = value
        previous = value

    for extra_key in ("phase_confidence", "impact_evidence", "wrist_peak_value"):
        if extra_key in phases:
            result[extra_key] = phases[extra_key]

    return result


# ============================================================
# PHASE SEGMENTATION / CONTACT
# ============================================================

def _smooth_signal(values, fps, window_seconds=0.06):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    valid = np.isfinite(arr)
    if valid.sum() == 0:
        return np.zeros_like(arr)
    idx = np.arange(arr.size)
    filled = arr.copy()
    filled[~valid] = np.interp(idx[~valid], idx[valid], arr[valid])
    window = max(5, int(window_seconds * max(fps, 1.0)))
    if window % 2 == 0:
        window += 1
    if window >= len(filled):
        window = max(3, len(filled) - 1 if len(filled) % 2 == 0 else len(filled))
    if window < 3:
        return filled
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(filled, kernel, mode="same")


def segment_phases(joint_series, wrist_speed, fps):
    """Improved temporal heuristic; bat/ball tracking is still not used."""
    n = len(joint_series)
    if n == 0:
        return {}
    fps = max(float(fps or 30.0), 1.0)

    ankle_x = np.full(n, np.nan)
    for i, joints in enumerate(joint_series):
        p = joints.get("right_ankle")
        if p and p[0] is not None:
            ankle_x[i] = float(p[0])
    if np.isfinite(ankle_x).sum() >= 2:
        idx = np.arange(n)
        ankle_x = np.interp(idx, idx[np.isfinite(ankle_x)], ankle_x[np.isfinite(ankle_x)])
        ankle_speed = np.abs(np.gradient(ankle_x) * fps)
    else:
        ankle_speed = np.zeros(n)
    ankle_smooth = _smooth_signal(ankle_speed, fps, 0.08)
    ankle_base = float(np.percentile(ankle_smooth, 50))
    ankle_p85 = float(np.percentile(ankle_smooth, 85))
    stride_thr = max(40.0, ankle_base + 0.35 * max(1.0, ankle_p85 - ankle_base))
    win = max(3, int(0.06 * fps))
    stride_start = 0
    for i in range(win, max(win + 1, n - win)):
        if np.mean(ankle_smooth[i-win+1:i+1]) >= stride_thr:
            stride_start = max(0, i - win // 2)
            break

    ws = np.asarray([np.nan if v is None else float(v) for v in wrist_speed], dtype=float)
    if ws.size < n:
        ws = np.pad(ws, (0, n-ws.size), constant_values=np.nan)
    elif ws.size > n:
        ws = ws[:n]
    if np.isfinite(ws).sum() == 0:
        ws = np.zeros(n)
    ws_smooth = _smooth_signal(ws, fps, 0.06)
    base = float(np.percentile(ws_smooth, 50))
    p90 = float(np.percentile(ws_smooth, 90))
    spread = max(1.0, p90 - base)
    down_thr = max(120.0, base + 0.55 * spread)
    search_start = min(n-1, max(stride_start + 1, int(0.03 * fps)))
    crossing = np.where(ws_smooth[search_start:] >= down_thr)[0]
    downswing_start = search_start + int(crossing[0]) if crossing.size else min(n-1, max(stride_start+1, int(0.2*n)))

    action_end = min(n-1, downswing_start + max(int(0.8*fps), int(0.25*n)))
    search = ws_smooth[downswing_start:action_end+1]
    if search.size:
        local=[]
        for k in range(1, max(1, search.size-1)):
            if search[k] >= search[k-1] and search[k] >= search[k+1]:
                local.append(k)
        peak_k = max(local, key=lambda k: float(search[k])) if local else int(np.argmax(search))
        impact_idx = downswing_start + peak_k
    else:
        impact_idx = None

    peak_value = float(ws_smooth[impact_idx]) if impact_idx is not None else 0.0
    prominence = float(np.clip((peak_value-base)/spread/2.0, 0.0, 1.0))
    settle = max(base + 0.15*spread, 0.35*peak_value)
    follow_end = min(n-1, (impact_idx if impact_idx is not None else downswing_start) + max(2, int(0.12*fps)))
    if impact_idx is not None:
        for k, v in enumerate(ws_smooth[impact_idx+1:]):
            if v <= settle and k+1 >= max(2, int(0.05*fps)):
                follow_end = impact_idx + 1 + k
                break
    recovery_start = min(n-1, follow_end+1)
    order_ok = stride_start <= downswing_start <= (impact_idx if impact_idx is not None else downswing_start) <= follow_end <= recovery_start
    duration_ok = max(3, int(0.08*fps)) <= max(1, follow_end-stride_start) <= max(5, int(2.0*fps))
    phase_conf = (0.45*prominence + 0.35*float(order_ok) + 0.20*float(duration_ok)) * 100.0
    return {
        "stance_start": 0,
        "stride_start": int(stride_start),
        "downswing_start": int(downswing_start),
        "impact_frame": int(impact_idx) if impact_idx is not None else None,
        "follow_through_end": int(follow_end),
        "recovery_start": int(recovery_start),
        "last_frame": n-1,
        "phase_confidence": round(phase_conf, 1),
        "impact_evidence": round(prominence*100.0, 1),
        "wrist_peak_value": round(peak_value, 2),
    }


def detect_contact_from_wrist_speed(wrist_speed, fps, phase_info=None):
    if phase_info and phase_info.get("impact_frame") is not None:
        idx=int(phase_info["impact_frame"])
        ws=np.asarray(wrist_speed,dtype=float)
        if 0 <= idx < len(ws) and np.isfinite(ws[idx]):
            return idx, float(ws[idx])
    if not wrist_speed:
        return None, None
    ws=np.asarray(wrist_speed,dtype=float)
    if not np.isfinite(ws).any():
        return None, None
    idx=int(np.nanargmax(ws))
    return idx, float(ws[idx])


def compute_analysis_confidence(reliability, phases, bat_angle_at_impact):
    pose=np.clip(float(reliability.get("overall_pose_coverage_pct",0.0))/90.0,0.0,1.0)
    wrist=np.clip(float(reliability.get("wrist_speed_coverage_pct",0.0))/90.0,0.0,1.0)
    phase=np.clip(float(phases.get("phase_confidence",0.0))/100.0,0.0,1.0)
    impact=np.clip(float(phases.get("impact_evidence",0.0))/100.0,0.0,1.0)
    bat=1.0 if bat_angle_at_impact is not None else 0.0
    score=(0.30*pose+0.20*wrist+0.25*phase+0.20*impact+0.05*bat)*100.0
    level="HIGH" if score>=80 else ("MEDIUM" if score>=60 else "LOW")
    return {
        "score": round(float(score),1),
        "level": level,
        "components": {
            "pose": round(pose*100,1),
            "wrist": round(wrist*100,1),
            "phase": round(phase*100,1),
            "impact": round(impact*100,1),
            "bat": round(bat*100,1),
        }
    }


# ============================================================
# SCORING
# ============================================================

def score_from_targets(
    value,
    lo,
    hi,
    slack=20.0
):
    if value is None:
        return 5

    if lo <= value <= hi:
        return 9

    mid = 0.5 * (lo + hi)

    return (
        7
        if abs(value - mid) <= slack
        else 5
    )


def grade_from_scores(scores):
    if not scores:
        return "N/A"

    avg = np.mean(
        list(scores.values())
    )

    if avg >= 8.0:
        return "Advanced"

    if avg >= 6.5:
        return "Intermediate"

    return "Beginner"


# ============================================================
# PDF REPORT
# ============================================================

def make_pdf_report(
    pdf_path,
    evaluation,
    png_plot_path,
    annotated_video_path=None
):

    c = canvas.Canvas(
        pdf_path,
        pagesize=A4
    )

    W, H = A4

    margin = 40

    c.setFont(
        "Helvetica-Bold",
        18
    )

    c.drawString(
        margin,
        H - margin,
        "VyronAI Cricket Analysis – Report"
    )

    y = H - margin - 30

    c.setFont(
        "Helvetica",
        11
    )

    c.drawString(
        margin,
        y,
        f"Video: {evaluation.get('video', '')}"
    )

    y -= 16

    c.drawString(
        margin,
        y,
        f"Average FPS: {evaluation.get('avg_fps', '-')}"
    )

    y -= 16

    c.drawString(
        margin,
        y,
        f"Skill Grade: {evaluation.get('skill_grade', '-')}"
    )

    y -= 16
    confidence = evaluation.get("analysis_confidence", {})
    c.drawString(
        margin,
        y,
        "Analysis Confidence: "
        f"{confidence.get('level', '-')} "
        f"({confidence.get('score', '-')}/100)"
    )

    y -= 22

    c.setFont(
        "Helvetica-Bold",
        12
    )

    c.drawString(
        margin,
        y,
        "Average Metrics"
    )

    y -= 14

    c.setFont(
        "Helvetica",
        11
    )

    averages = evaluation.get(
        "averages",
        {}
    )

    metric_keys = [
        "elbow_angle_deg",
        "spine_lean_deg",
        "head_knee_diff_px",
        "foot_direction_deg",
        "bat_angle_deg_at_impact"
    ]

    for key in metric_keys:

        value = averages.get(key)

        display_value = (
            round(value, 2)
            if value is not None
            else "-"
        )

        c.drawString(
            margin,
            y,
            f"• {key}: {display_value}"
        )

        y -= 14

    y -= 8

    c.setFont(
        "Helvetica-Bold",
        12
    )

    c.drawString(
        margin,
        y,
        "Scores"
    )

    y -= 14

    c.setFont(
        "Helvetica",
        11
    )

    for key, value in evaluation.get(
        "scores",
        {}
    ).items():

        c.drawString(
            margin,
            y,
            f"• {key}: {value}"
        )

        y -= 14

    y -= 8

    c.setFont(
        "Helvetica-Bold",
        12
    )

    c.drawString(
        margin,
        y,
        "Phases (frames)"
    )

    y -= 14

    c.setFont(
        "Helvetica",
        11
    )

    for key, value in evaluation.get(
        "phases",
        {}
    ).items():

        c.drawString(
            margin,
            y,
            f"• {key}: {value}"
        )

        y -= 14

    y -= 8

    if os.path.exists(
        png_plot_path
    ):

        try:

            img = ImageReader(
                png_plot_path
            )

            ih = 220

            iw = W - 2 * margin

            c.drawImage(
                img,
                margin,
                y - ih,
                width=iw,
                height=ih,
                preserveAspectRatio=True,
                mask="auto"
            )

            y -= ih + 10

        except Exception:
            pass

    c.setFont(
        "Helvetica-Oblique",
        9
    )

    c.setFillColor(
        colors.grey
    )

    c.drawString(
        margin,
        margin,
        "Generated by VyronAI"
    )

    c.save()


# ============================================================
# MAIN VIDEO ANALYSIS
# ============================================================

def analyze_video(
    video_path: str
) -> dict:

    targets = ensure_targets_config()

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video: {video_path}"
        )

    fps = (
        cap.get(
            cv2.CAP_PROP_FPS
        )
        or
        25.0
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    # --------------------------------------------------------
    # Video writer
    # --------------------------------------------------------

    writer = cv2.VideoWriter(
        ANNOTATED_VIDEO_PATH,
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        fps,
        (width, height)
    )

    if not writer.isOpened():

        # Fallback
        writer = cv2.VideoWriter(
            ANNOTATED_VIDEO_PATH,
            cv2.VideoWriter_fourcc(
                *"MJPG"
            ),
            fps,
            (width, height)
        )

    metrics = []

    elbow_list = []
    spine_list = []
    time_list = []

    frame_times = []

    wrist_speed_series = []

    joint_series = []

    bat_angles = []
    bat_lines = []

    # New Ball / Bat tracking series.
    ball_x_series = []
    ball_y_series = []
    ball_conf_series = []

    bat_x_series = []
    bat_y_series = []
    bat_conf_series = []

    ball_bat_distance_series = []

    ball_detected_count = 0
    bat_detected_count = 0
    distance_count = 0

    ball_bat_model = None

    if ENABLE_BALL_BAT_TRACKING:
        try:
            ball_bat_model = load_ball_bat_model()
        except Exception as model_exc:
            print(
                "WARNING: Ball/Bat model unavailable. "
                f"{model_exc}"
            )
            ball_bat_model = None

    frame_idx = 0

    # --------------------------------------------------------
    # Current MediaPipe PoseLandmarker
    # --------------------------------------------------------

    try:

        pose_landmarker = create_pose_landmarker()

    except Exception as exc:

        cap.release()
        writer.release()

        raise RuntimeError(
            "MediaPipe PoseLandmarker could not be created.\n"
            f"{exc}"
        ) from exc


    try:

        while True:

            t0 = time.time()

            ret, frame = cap.read()

            if not ret:
                break

            frame_idx += 1

            # BGR -> RGB
            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp_ms = int(
                (frame_idx / fps)
                * 1000
            )

            result = (
                pose_landmarker
                .detect_for_video(
                    mp_image,
                    timestamp_ms
                )
            )

            joints = {}

            elbow_deg = None
            spine_deg = None
            headknee_px = None
            footdir_deg = None

            bat_deg = None
            bat_line = None


            # ------------------------------------------------
            # Pose result
            # ------------------------------------------------

            if result.pose_landmarks:

                # First / primary pose
                landmarks = result.pose_landmarks[0]

                joints = get_joints_from_landmarks(
                    landmarks,
                    width,
                    height
                )

                required_keys = {
                    "right_shoulder",
                    "right_elbow",
                    "right_wrist",
                    "left_wrist",
                    "right_hip",
                    "right_knee",
                    "right_ankle",
                    "nose"
                }

                if required_keys.issubset(
                    joints.keys()
                ):

                    try:

                        elbow_deg = angle_ABC(
                            joints["right_shoulder"],
                            joints["right_elbow"],
                            joints["right_wrist"]
                        )

                        if elbow_deg is not None:
                            elbow_deg = round(
                                elbow_deg,
                                1
                            )


                        spine_deg = spine_lean_deg(
                            joints["right_hip"],
                            joints["right_shoulder"]
                        )


                        headknee_px = head_knee_diff_px(
                            joints["nose"],
                            joints["right_knee"]
                        )

                        if headknee_px is not None:
                            headknee_px = round(
                                headknee_px,
                                1
                            )


                        footdir_deg = foot_direction_deg(
                            joints["right_ankle"],
                            joints["right_knee"]
                        )

                        if footdir_deg is not None:
                            footdir_deg = round(
                                footdir_deg,
                                1
                            )


                        bat_deg, bat_line = estimate_bat_line(
                            frame,
                            joints
                        )

                        if bat_deg is not None:
                            bat_deg = round(
                                bat_deg,
                                1
                            )

                    except Exception:
                        pass


                # Draw pose
                draw_pose_landmarks(
                    frame,
                    landmarks
                )



            # ------------------------------------------------
            # Ball / Bat ROI tracking
            # ------------------------------------------------

            ball_detection = None
            bat_detection = None
            ball_roi = None
            bat_roi = None

            if (
                ENABLE_BALL_BAT_TRACKING
                and
                ball_bat_model is not None
            ):
                try:
                    if joints:
                        bat_detection, bat_roi = (
                            detect_bat_in_batsman_roi(
                                frame,
                                joints,
                                ball_bat_model
                            )
                        )

                        ball_detection, ball_roi = (
                            detect_ball_in_broad_roi(
                                frame,
                                joints,
                                ball_bat_model
                            )
                        )

                except Exception as tracking_exc:
                    # Keep the original pose pipeline running even if
                    # object detection fails for a frame.
                    ball_detection = None
                    bat_detection = None

            if ball_detection is not None:
                ball_x_series.append(
                    round(
                        float(
                            ball_detection["center_x"]
                        ),
                        2
                    )
                )
                ball_y_series.append(
                    round(
                        float(
                            ball_detection["center_y"]
                        ),
                        2
                    )
                )
                ball_conf_series.append(
                    round(
                        float(
                            ball_detection["confidence"]
                        ),
                        4
                    )
                )
                ball_detected_count += 1

            else:
                ball_x_series.append(None)
                ball_y_series.append(None)
                ball_conf_series.append(None)

            if bat_detection is not None:
                bat_x_series.append(
                    round(
                        float(
                            bat_detection["center_x"]
                        ),
                        2
                    )
                )
                bat_y_series.append(
                    round(
                        float(
                            bat_detection["center_y"]
                        ),
                        2
                    )
                )
                bat_conf_series.append(
                    round(
                        float(
                            bat_detection["confidence"]
                        ),
                        4
                    )
                )
                bat_detected_count += 1

            else:
                bat_x_series.append(None)
                bat_y_series.append(None)
                bat_conf_series.append(None)

            distance = ball_bat_distance(
                ball_detection,
                bat_detection
            )

            ball_bat_distance_series.append(
                round(distance, 2)
                if distance is not None
                else None
            )

            if distance is not None:
                distance_count += 1

            # Draw the new detections.
            draw_detection(
                frame,
                ball_detection,
                "BALL",
                (0, 0, 255)
            )

            draw_detection(
                frame,
                bat_detection,
                "BAT",
                (0, 255, 255)
            )

            if distance is not None:
                cv2.putText(
                    frame,
                    f"Ball-Bat: {distance:.1f}px",
                    (24, y + 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2
                )

            # ------------------------------------------------
            # Temporal series
            # ------------------------------------------------

            joint_series.append(
                joints
            )

            if (
                len(joint_series) >= 2
                and
                joints.get("right_wrist")
                and
                joint_series[-2].get("right_wrist")
            ):

                x2, y2 = joints[
                    "right_wrist"
                ]

                x1, y1 = joint_series[-2][
                    "right_wrist"
                ]

                wrist_speed_series.append(
                    math.hypot(
                        x2 - x1,
                        y2 - y1
                    )
                    *
                    fps
                )

            else:

                wrist_speed_series.append(
                    0.0
                )


            elbow_list.append(
                elbow_deg
            )

            spine_list.append(
                spine_deg
            )

            time_list.append(
                frame_idx / fps
            )

            bat_angles.append(
                bat_deg
            )

            bat_lines.append(
                bat_line
            )


            metrics.append({
                "frame": frame_idx,
                "elbow_angle": elbow_deg,
                "spine_lean": spine_deg,
                "head_knee_diff": headknee_px,
                "foot_direction_angle": footdir_deg,
                "bat_angle": bat_deg,

                # New object-tracking fields.
                "ball_x": ball_x_series[-1],
                "ball_y": ball_y_series[-1],
                "ball_confidence": ball_conf_series[-1],

                "bat_yolo_x": bat_x_series[-1],
                "bat_yolo_y": bat_y_series[-1],
                "bat_yolo_confidence": bat_conf_series[-1],

                "ball_bat_distance_px":
                    ball_bat_distance_series[-1],

                "joints": joints
            })


            # ------------------------------------------------
            # Threshold colouring
            # ------------------------------------------------

            def in_range(value, lo, hi):

                return (
                    value is not None
                    and
                    lo <= value <= hi
                )


            c_elb = (
                (0, 255, 0)
                if in_range(
                    elbow_deg,
                    *targets["elbow_angle_deg"]
                )
                else
                (0, 0, 255)
            )

            c_spi = (
                (0, 255, 0)
                if in_range(
                    spine_deg,
                    *targets["spine_lean_deg"]
                )
                else
                (0, 0, 255)
            )

            c_hk = (
                (0, 255, 0)
                if in_range(
                    headknee_px,
                    *targets["head_knee_diff_px"]
                )
                else
                (0, 0, 255)
            )

            c_foot = (
                (0, 255, 0)
                if in_range(
                    footdir_deg,
                    *targets["foot_direction_deg"]
                )
                else
                (0, 0, 255)
            )


            # ------------------------------------------------
            # Metrics overlay
            # ------------------------------------------------

            y = 36

            overlay_items = [
                (
                    f"Elbow: "
                    f"{elbow_deg if elbow_deg is not None else '-'}",
                    c_elb
                ),
                (
                    f"Spine: "
                    f"{spine_deg if spine_deg is not None else '-'}",
                    c_spi
                ),
                (
                    f"Head-Knee: "
                    f"{headknee_px if headknee_px is not None else '-'}",
                    c_hk
                ),
                (
                    f"Foot Dir: "
                    f"{footdir_deg if footdir_deg is not None else '-'}",
                    c_foot
                ),
                (
                    f"Bat: "
                    f"{bat_deg if bat_deg is not None else '-'}",
                    (255, 255, 0)
                )
            ]


            for text, color in overlay_items:

                cv2.putText(
                    frame,
                    text,
                    (24, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    3
                )

                cv2.putText(
                    frame,
                    text,
                    (24, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

                y += 28


            # ------------------------------------------------
            # Bat line
            # ------------------------------------------------

            if bat_line is not None:

                xa, ya, xb, yb = bat_line

                cv2.line(
                    frame,
                    (xa, ya),
                    (xb, yb),
                    (0, 255, 255),
                    2
                )


            # ------------------------------------------------
            # FPS overlay
            # ------------------------------------------------

            dt = time.time() - t0

            frame_times.append(
                dt
            )

            inst_fps = (
                1.0 / dt
                if dt > 0
                else 0.0
            )

            cv2.putText(
                frame,
                f"FPS: {inst_fps:.1f}",
                (width - 160, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                3
            )

            cv2.putText(
                frame,
                f"FPS: {inst_fps:.1f}",
                (width - 160, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )


            # ------------------------------------------------
            # Write
            # ------------------------------------------------

            writer.write(
                frame
            )

    finally:

        cap.release()
        writer.release()

        try:
            pose_landmarker.close()
        except Exception:
            pass

        cv2.destroyAllWindows()


    # ========================================================
    # POST-PROCESSING
    # ========================================================

    avg_fps = (
        len(frame_times)
        /
        (
            sum(frame_times)
            +
            1e-9
        )
    )


    # --------------------------------------------------------
    # Phases / contact
    # --------------------------------------------------------

    phases = segment_phases(
        joint_series,
        wrist_speed_series,
        fps
    )

    # Safe chronological view for the new graph / metrics.
    phases = clean_phase_order(
        phases,
        len(joint_series),
    )

    impact_frame_auto, peak_ws = (
        detect_contact_from_wrist_speed(
            wrist_speed_series,
            fps,
            phase_info=phases,
        )
    )


    # --------------------------------------------------------
    # Smoothness
    # --------------------------------------------------------

    elbow_vals, elbow_deltas, elbow_var = (
        compute_smoothness(
            elbow_list
        )
    )

    spine_vals, spine_deltas, spine_var = (
        compute_smoothness(
            spine_list
        )
    )


    # --------------------------------------------------------
    # Graphs
    # --------------------------------------------------------

    # Legacy graph is preserved but hidden from the new UI.
    if SHOW_LEGACY_GRAPH:
        plt.figure(
            figsize=(10, 4)
        )

        plt.plot(
            time_list,
            elbow_vals,
            label="Elbow angle (deg)"
        )

        plt.plot(
            time_list,
            spine_vals,
            label="Spine lean (deg)"
        )

        plt.xlabel("Time (s)")
        plt.ylabel("Angle (deg)")
        plt.title(
            "Temporal Trends: Elbow & Spine (Legacy)"
        )

        plt.legend(
            loc="best"
        )

        plt.tight_layout()

        plt.savefig(
            LEGACY_GRAPH_PNG_PATH
        )

        plt.close()

    # New graph: no forward-filling + wrist-speed + phase markers.
    if SHOW_NEW_GRAPH:
        plot_clean_movement_analysis(
            time_list=time_list,
            elbow_series=elbow_list,
            spine_series=spine_list,
            wrist_speed_series=wrist_speed_series,
            phases=phases,
            fps=fps,
            output_path=NEW_GRAPH_PNG_PATH,
        )


    # --------------------------------------------------------
    # Bat path / impact angle
    # --------------------------------------------------------

    straightness = path_straightness(
        bat_angles
    )

    bat_angle_at_impact = None

    if (
        impact_frame_auto is not None
        and
        0 <= impact_frame_auto < len(bat_angles)
    ):

        bat_angle_at_impact = (
            bat_angles[
                impact_frame_auto
            ]
        )


    # --------------------------------------------------------
    # Per-frame CSV
    # --------------------------------------------------------

    df = pd.DataFrame(
        metrics
    )

    df.to_csv(
        PER_FRAME_CSV_PATH,
        index=False
    )


    # --------------------------------------------------------
    # Averages
    # --------------------------------------------------------

    def avg_nonan(values):

        arr = np.array(
            [
                value
                for value in values
                if value is not None
            ],
            float
        )

        return (
            float(arr.mean())
            if arr.size
            else None
        )


    avg_elbow = avg_nonan(
        [
            m["elbow_angle"]
            for m in metrics
        ]
    )

    avg_spine = avg_nonan(
        [
            m["spine_lean"]
            for m in metrics
        ]
    )

    avg_hk = avg_nonan(
        [
            m["head_knee_diff"]
            for m in metrics
        ]
    )

    avg_foot = avg_nonan(
        [
            m["foot_direction_angle"]
            for m in metrics
        ]
    )

    # --------------------------------------------------------
    # Pose reliability
    # --------------------------------------------------------

    elbow_valid, elbow_total, elbow_coverage = compute_validity(
        elbow_list
    )
    spine_valid, spine_total, spine_coverage = compute_validity(
        spine_list
    )

    head_valid, head_total, head_coverage = compute_validity(
        [
            m["head_knee_diff"]
            for m in metrics
        ]
    )

    foot_valid, foot_total, foot_coverage = compute_validity(
        [
            m["foot_direction_angle"]
            for m in metrics
        ]
    )

    wrist_valid, wrist_total, wrist_coverage = compute_validity(
        [
            v if v != 0.0 else None
            for v in wrist_speed_series
        ]
    )

    overall_pose_coverage = round(
        np.mean([elbow_coverage, spine_coverage, head_coverage, foot_coverage]),
        1,
    )

    reliability = {
        "overall_pose_coverage_pct": overall_pose_coverage,
        "elbow_coverage_pct": elbow_coverage,
        "spine_coverage_pct": spine_coverage,
        "head_knee_coverage_pct": head_coverage,
        "foot_coverage_pct": foot_coverage,
        "wrist_speed_coverage_pct": wrist_coverage,

        # Ball/Bat tracking quality.
        "ball_detection_coverage_pct": round(
            (ball_detected_count / max(1, frame_idx)) * 100.0,
            1
        ),
        "bat_detection_coverage_pct": round(
            (bat_detected_count / max(1, frame_idx)) * 100.0,
            1
        ),
        "ball_bat_distance_coverage_pct": round(
            (distance_count / max(1, frame_idx)) * 100.0,
            1
        ),
    }

    analysis_confidence = compute_analysis_confidence(
        reliability, phases, bat_angle_at_impact
    )

    scores = {
        "Footwork": score_from_targets(
            avg_foot,
            *DEFAULT_TARGETS[
                "foot_direction_deg"
            ]
        ),

        "Head Position": score_from_targets(
            avg_hk,
            *DEFAULT_TARGETS[
                "head_knee_diff_px"
            ]
        ),

        "Swing Control": score_from_targets(
            avg_elbow,
            *DEFAULT_TARGETS[
                "elbow_angle_deg"
            ]
        ),

        "Balance": score_from_targets(
            avg_spine,
            *DEFAULT_TARGETS[
                "spine_lean_deg"
            ]
        ),

        "Follow-through": 8
    }


    legacy_skill_grade = grade_from_scores(scores)
    grade = legacy_skill_grade if analysis_confidence["score"] >= 60.0 else "Low-confidence analysis"

    evaluation = {
        "video": os.path.basename(
            video_path
        ),

        "avg_fps": round(
            avg_fps,
            2
        ),

        "phases": phases,

        "contact_detection": {
            "impact_frame_auto": impact_frame_auto,
            "peak_wrist_speed": peak_ws
        },

        "averages": {
            "elbow_angle_deg": avg_elbow,
            "spine_lean_deg": avg_spine,
            "head_knee_diff_px": avg_hk,
            "foot_direction_deg": avg_foot,
            "bat_angle_deg_at_impact":
                bat_angle_at_impact,
            "bat_path_straightness_0to1":
                straightness
        },

        "scores": scores,

        "skill_grade": grade,
        "legacy_skill_grade": legacy_skill_grade,
        "analysis_confidence": analysis_confidence,

        "reliability": reliability,

        "tracking": {
            "model_path": CUSTOM_MODEL_PATH,
            "bat_class_id": BAT_CLASS_ID,
            "ball_class_id": BALL_CLASS_ID,
            "ball_detection_coverage_pct":
                reliability["ball_detection_coverage_pct"],
            "bat_detection_coverage_pct":
                reliability["bat_detection_coverage_pct"],
            "ball_bat_distance_coverage_pct":
                reliability["ball_bat_distance_coverage_pct"],
        },


        "artifacts": {
            "annotated_video":
                ANNOTATED_VIDEO_PATH,
            "legacy_graph":
                LEGACY_GRAPH_PNG_PATH,
            "new_movement_graph":
                NEW_GRAPH_PNG_PATH,
            "per_frame_csv":
                PER_FRAME_CSV_PATH,
            "report_pdf":
                PDF_REPORT_PATH
        }
    }


    # ========================================================
    # SAVE EVALUATION JSON
    # ========================================================

    if os.path.exists(
        EVAL_JSON_PATH
    ):

        try:

            with open(
                EVAL_JSON_PATH,
                "r",
                encoding="utf-8"
            ) as f:

                data = json.load(f)

            if not isinstance(
                data,
                list
            ):
                data = [data]

        except Exception:

            data = []

    else:

        data = []


    data.append(
        evaluation
    )


    with open(
        EVAL_JSON_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=4
        )


    # ========================================================
    # PDF
    # ========================================================

    make_pdf_report(
        PDF_REPORT_PATH,
        evaluation,
        NEW_GRAPH_PNG_PATH,
        ANNOTATED_VIDEO_PATH
    )


    return evaluation