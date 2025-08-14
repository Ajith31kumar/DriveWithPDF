# backend_bonus.py
import os, json, time, math, cv2, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
import mediapipe as mp
from pathlib import Path

# -----------------------------
# Paths / I/O
# -----------------------------
OUTPUT_DIR = "output"
CONFIG_DIR = "config"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)

ANNOTATED_VIDEO_PATH   = os.path.join(OUTPUT_DIR, "annotated_bonus.mp4")
# SMOOTH_GRAPH_VIDEO     = os.path.join(OUTPUT_DIR, "smoothness_graph.mp4")
# SMOOTHNESS_PNG_PATH    = os.path.join(OUTPUT_DIR, "smoothness.png")
FINAL_GRAPH_PNG_PATH   = os.path.join(OUTPUT_DIR, "smoothness_final.png")
EVAL_JSON_PATH         = os.path.join(OUTPUT_DIR, "evaluation.json")
PER_FRAME_CSV_PATH     = os.path.join(OUTPUT_DIR, "per_frame_metrics.csv")
PDF_REPORT_PATH        = os.path.join(OUTPUT_DIR, "report.pdf")

TARGETS_JSON_PATH      = os.path.join(CONFIG_DIR, "targets.json")

# -----------------------------
# Default targets (reference comparison)
# -----------------------------
DEFAULT_TARGETS = {
    "elbow_angle_deg":        [120, 170],
    "spine_lean_deg":         [0,   20],
    "head_knee_diff_px":      [0,   40],
    "foot_direction_deg":     [-45, 45],
    "bat_angle_deg_at_impact":[-20, 20]
}

def ensure_targets_config(path=TARGETS_JSON_PATH):
    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump(DEFAULT_TARGETS, f, indent=4)
    with open(path, "r") as f:
        return json.load(f)

# -----------------------------
# Pose / Drawing
# -----------------------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# -----------------------------
# Metric helpers
# -----------------------------
def angle_ABC(a, b, c):
    a, b, c = np.array(a, float), np.array(b, float), np.array(c, float)
    if (np.linalg.norm(a-b) < 1e-6) or (np.linalg.norm(c-b) < 1e-6):
        return None
    ba, bc = a-b, c-b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc))
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def spine_lean_deg(hip, shoulder):
    # hip, shoulder: (x,y) pixels
    dx = shoulder[0] - hip[0]
    dy = shoulder[1] - hip[1]
    # angle measured from vertical axis (y): use arctan2(dx, dy)
    angle = float(np.degrees(np.arctan2(dx, dy)))  # could be -180..180
    # Make absolute tilt relative to vertical (0..180)
    angle = abs(angle)
    # Normalize to 0..90 (a tilt of 170° -> 10° from vertical)
    if angle > 90:
        angle = 180 - angle
    return round(angle, 1)


def head_knee_diff_px(nose, knee):
    return float(abs(nose[0] - knee[0]))

def foot_direction_deg(ankle, knee):
    dx = knee[0] - ankle[0]
    dy = knee[1] - ankle[1]
    return float(np.degrees(np.arctan2(dy, dx)))

def safe_get_landmark_px(lms, lid, w, h):
    lm = lms[lid]
    return (int(lm.x*w), int(lm.y*h))

# -----------------------------
# Basic bat detection/tracking
# -----------------------------
def estimate_bat_line(frame_bgr, joints, box=120):
    """
    ROI around midpoint between wrists → Canny → HoughLinesP.
    Returns (angle_deg, (xA,yA,xB,yB)) or (None, None)
    """
    if not ("right_wrist" in joints and "left_wrist" in joints):
        return None, None
    (x1, y1), (x2, y2) = joints["right_wrist"], joints["left_wrist"]
    cx, cy = (x1+x2)//2, (y1+y2)//2
    h, w = frame_bgr.shape[:2]
    x0, y0 = max(0, cx-box), max(0, cy-box)
    x1, y1 = min(w, cx+box), min(h, cy+box)
    roi = frame_bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return None, None
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 180)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=30, maxLineGap=10)
    if lines is None:
        return None, None
    best = None
    best_len = 0
    for l in lines:
        xa, ya, xb, yb = l[0]
        length = (xb-xa)**2 + (yb-ya)**2
        if length > best_len:
            best_len = length
            best = (xa, ya, xb, yb)
    if best is None:
        return None, None
    xa, ya, xb, yb = best
    # convert back to full-frame coords
    xa, ya, xb, yb = xa+x0, ya+y0, xb+x0, yb+y0
    ang = float(np.degrees(np.arctan2((yb-ya), (xb-xa))))
    return ang, (xa, ya, xb, yb)

def path_straightness(angles_deg):
    """1 - normalized variance as a crude 'straightness' score [0..1]."""
    arr = np.array([a for a in angles_deg if a is not None], float)
    if arr.size < 5:  # not enough
        return None
    var = float(np.var(arr))
    # map variance to [0..1] inversely (tune scale)
    return float(np.clip(1.0 / (1.0 + var/400.0), 0.0, 1.0))

# -----------------------------
# Smoothness & chart
# -----------------------------
def compute_smoothness(series):
    vals = np.array([np.nan if v is None else float(v) for v in series], float)
    for i in range(1, len(vals)):
        if np.isnan(vals[i]): vals[i] = vals[i-1]
    if len(vals) and np.isnan(vals[0]): vals[0] = 0.0
    deltas = np.abs(np.diff(vals)) if len(vals) > 1 else np.array([])
    variance = float(np.nanvar(vals)) if len(vals) else 0.0
    return vals, deltas, variance



# -----------------------------
# Phase segmentation & contact detection
# -----------------------------
def segment_phases(joint_series, wrist_speed, fps):
    """
    Heuristic segmentation:
    - Stride start: ankle horizontal speed crosses a threshold
    - Downswing start: wrist speed rises past 75th percentile
    - Impact: wrist speed max
    - Follow-through end: wrist speed drops below median after impact
    """
    n = len(joint_series)
    if n == 0:
        return {}
    x_ank = [j.get("right_ankle", (None, None))[0] for j in joint_series]
    # fill missing
    for i in range(1, n):
        if x_ank[i] is None:
            x_ank[i] = x_ank[i-1]
    if x_ank[0] is None: x_ank[0] = 0

    vx = [0.0] + [ (x_ank[i]-x_ank[i-1]) * fps for i in range(1, n) ]

    stride_start = 0
    SPEED_THR = 60.0
    win = max(3, int(0.05*fps))
    for i in range(win, n):
        if np.mean(np.abs(vx[i-win+1:i+1])) > SPEED_THR:
            stride_start = max(0, i - win//2)
            break

    ws = np.array(wrist_speed + [0.0]*(n-len(wrist_speed)))
    ramp_thr = max(150.0, float(np.percentile(ws, 75))) if ws.size else 150.0
    downswing_start = int(np.argmax(ws > ramp_thr))
    if ws.size == 0 or ws[downswing_start] <= ramp_thr:
        downswing_start = max(stride_start+1, int(0.2*n))

    impact_idx = int(np.argmax(ws)) if ws.size else None

    settle_thr = float(np.median(ws)) if ws.size else 0.0
    follow_end = impact_idx if impact_idx is not None else n//2
    if impact_idx is not None:
        for i in range(impact_idx+1, n):
            if ws[i] < settle_thr:
                follow_end = i
                break
    recovery_start = min(follow_end+1, n-1)

    return {
        "stance_start": 0,
        "stride_start": int(stride_start),
        "downswing_start": int(downswing_start),
        "impact_frame": int(impact_idx) if impact_idx is not None else None,
        "follow_through_end": int(follow_end),
        "recovery_start": int(recovery_start),
        "last_frame": n-1
    }

def detect_contact_from_wrist_speed(wrist_speed, fps):
    """Return (impact_frame, peak_speed)."""
    if not wrist_speed:
        return None, None
    ws = np.array(wrist_speed, float)
    idx = int(np.argmax(ws))
    return idx, float(ws[idx])

# -----------------------------
# Scoring / Grade
# -----------------------------
def score_from_targets(value, lo, hi, slack=20.0):
    if value is None: return 5
    if lo <= value <= hi: return 9
    mid = 0.5*(lo+hi)
    return 7 if abs(value-mid) <= slack else 5

def grade_from_scores(scores):
    avg = np.mean(list(scores.values()))
    if avg >= 8.0: return "Advanced"
    if avg >= 6.5: return "Intermediate"
    return "Beginner"
def normalize_angle_deg(angle):
    """Return angle in -180..180"""
    if angle is None:
        return None
    a = ((angle + 180) % 360) - 180
    return round(a, 1)
def normalized_head_knee(nose, knee, hip):
    # returns head-knee horizontal diff as proportion of torso height
    torso_h = math.hypot(nose[0]-hip[0], nose[1]-hip[1])
    if torso_h < 1e-6:
        return None
    raw_px = abs(nose[0] - knee[0])
    return round((raw_px / torso_h) * 100.0, 1)  # percent of torso height

# -----------------------------
# PDF Report
# -----------------------------
def make_pdf_report(pdf_path, evaluation, png_plot_path, annotated_video_path=None):
    c = canvas.Canvas(pdf_path, pagesize=A4)
    W, H = A4
    margin = 40

    # Title
    c.setFont("Helvetica-Bold", 18)
    c.drawString(margin, H - margin, "AthleteRise Cover Drive Analysis – Report")

    y = H - margin - 30
    # Summary block
    c.setFont("Helvetica", 11)
    c.drawString(margin, y, f"Video: {evaluation.get('video','')}")
    y -= 16
    c.drawString(margin, y, f"Average FPS: {evaluation.get('avg_fps','-')}")
    y -= 16
    c.drawString(margin, y, f"Skill Grade: {evaluation.get('skill_grade','-')}")
    y -= 22

    # Averages
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Average Metrics")
    y -= 14
    c.setFont("Helvetica", 11)
    av = evaluation.get("averages", {})
    for k in ["elbow_angle_deg","spine_lean_deg","head_knee_diff_px","foot_direction_deg","bat_angle_deg_at_impact"]:
        c.drawString(margin, y, f"• {k}: {round(av[k],2) if av.get(k) is not None else '-'}")
        y -= 14

    # Scores
    y -= 8
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Scores")
    y -= 14
    c.setFont("Helvetica", 11)
    for k, v in evaluation.get("scores", {}).items():
        c.drawString(margin, y, f"• {k}: {v}")
        y -= 14

    # Phases
    y -= 8
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Phases (frames)")
    y -= 14
    c.setFont("Helvetica", 11)
    for k, v in evaluation.get("phases", {}).items():
        c.drawString(margin, y, f"• {k}: {v}")
        y -= 14

    # Plot image
    y -= 8
    if os.path.exists(png_plot_path):
        try:
            img = ImageReader(png_plot_path)
            # reserve area
            ih = 220
            iw = W - 2*margin
            c.drawImage(img, margin, y-ih, width=iw, height=ih, preserveAspectRatio=True, mask='auto')
            y -= ih + 10
        except Exception:
            pass

    # Footer
    c.setFont("Helvetica-Oblique", 9)
    c.setFillColor(colors.grey)
    c.drawString(margin, margin, "Generated by AthleteRise Analyzer")
    c.save()

# -----------------------------
# Main analysis
# -----------------------------
def analyze_video(video_path: str) -> dict:
    targets = ensure_targets_config()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Writers for two separate videos
    annotated_writer = cv2.VideoWriter(ANNOTATED_VIDEO_PATH, cv2.VideoWriter_fourcc(*"avc1"), fps, (width, height))
    # graph_writer     = cv2.VideoWriter(SMOOTH_GRAPH_VIDEO,   cv2.VideoWriter_fourcc(*"avc1"), fps, (width, height))

    metrics = []
    elbow_list, spine_list, time_list = [], [], []
    frame_times = []
    wrist_speed_series = []
    joint_series = []
    bat_angles = []
    bat_lines  = []   # list of line segments for optional debug/straightness

    frame_idx = 0
    with mp_pose.Pose(static_image_mode=False,
                      model_complexity=1,
                      enable_segmentation=False,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            joints = {}
            elbow_deg = spine_deg = headknee_px = footdir_deg = None
            bat_deg = None
            bat_line = None

            if results.pose_landmarks:
                lms = results.pose_landmarks.landmark
                ids = mp_pose.PoseLandmark
                try:
                    req = {
                        "nose": ids.NOSE,
                        "right_shoulder": ids.RIGHT_SHOULDER,
                        "right_elbow": ids.RIGHT_ELBOW,
                        "right_wrist": ids.RIGHT_WRIST,
                        "left_wrist": ids.LEFT_WRIST,
                        "right_hip": ids.RIGHT_HIP,
                        "right_knee": ids.RIGHT_KNEE,
                        "right_ankle": ids.RIGHT_ANKLE
                    }
                    for k, lid in req.items():
                        joints[k] = safe_get_landmark_px(lms, lid, width, height)

                    elbow_deg = angle_ABC(joints["right_shoulder"], joints["right_elbow"], joints["right_wrist"])
                    if elbow_deg is not None: elbow_deg = round(elbow_deg, 1)

                    spine_deg = spine_lean_deg(joints["right_hip"], joints["right_shoulder"])
                    if spine_deg is not None: spine_deg = round(spine_deg, 1)

                    headknee_px = head_knee_diff_px(joints["nose"], joints["right_knee"])
                    if headknee_px is not None: headknee_px = round(headknee_px, 1)

                    footdir_deg = foot_direction_deg(joints["right_ankle"], joints["right_knee"])
                    if footdir_deg is not None: footdir_deg = round(footdir_deg, 1)

                    # Bat line
                    bat_deg, bat_line = estimate_bat_line(frame, joints)
                    if bat_deg is not None: bat_deg = round(bat_deg, 1)

                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                except Exception:
                    pass

            # series
            joint_series.append(joints)
            if len(joint_series) >= 2 and joints.get("right_wrist") and joint_series[-2].get("right_wrist"):
                (x2,y2) = joints["right_wrist"]; (x1,y1) = joint_series[-2]["right_wrist"]
                wrist_speed_series.append(math.hypot(x2-x1, y2-y1) * fps)
            else:
                wrist_speed_series.append(0.0)

            elbow_list.append(elbow_deg)
            spine_list.append(spine_deg)
            time_list.append(frame_idx / fps)
            bat_angles.append(bat_deg)
            bat_lines.append(bat_line)

            metrics.append({
                "frame": frame_idx,
                "elbow_angle": elbow_deg,
                "spine_lean": spine_deg,
                "head_knee_diff": headknee_px,
                "foot_direction_angle": footdir_deg,
                "bat_angle": bat_deg,
                "joints": joints
            })

            # Live thresholds coloring
            def in_range(v, lo, hi): return (v is not None) and (lo <= v <= hi)
            c_elb = (0,255,0) if in_range(elbow_deg, *targets["elbow_angle_deg"]) else (0,0,255)
            c_spi = (0,255,0) if in_range(spine_deg, *targets["spine_lean_deg"]) else (0,0,255)
            c_hk  = (0,255,0) if in_range(headknee_px, *targets["head_knee_diff_px"]) else (0,0,255)
            c_foot= (0,255,0) if in_range(footdir_deg, *targets["foot_direction_deg"]) else (0,0,255)

            y = 36
            for text, color in [
                (f"Elbow: {elbow_deg if elbow_deg is not None else '-'}", c_elb),
                (f"Spine: {spine_deg if spine_deg is not None else '-'}", c_spi),
                (f"Head-Knee: {headknee_px if headknee_px is not None else '-'}", c_hk),
                (f"Foot Dir: {footdir_deg if footdir_deg is not None else '-'}", c_foot),
                (f"Bat: {bat_deg if bat_deg is not None else '-'}", (255,255,0)),
            ]:
                cv2.putText(frame, text, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3)
                cv2.putText(frame, text, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                y += 28

            # draw bat line if detected
            if bat_line is not None:
                xa, ya, xb, yb = bat_line
                cv2.line(frame, (xa,ya), (xb,yb), (0,255,255), 2)

            # FPS overlay
            dt = time.time() - t0
            frame_times.append(dt)
            inst_fps = 1.0 / dt if dt > 0 else 0.0
            cv2.putText(frame, f"FPS: {inst_fps:.1f}", (width-160, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3)
            cv2.putText(frame, f"FPS: {inst_fps:.1f}", (width-160, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            # write annotated
            annotated_writer.write(frame)

    cap.release()
    annotated_writer.release()
    cv2.destroyAllWindows()

    # -------- Post-processing --------
    avg_fps = len(frame_times) / (sum(frame_times) + 1e-9)

    # Phases + Contact
    phases = segment_phases(joint_series, wrist_speed_series, fps)
    impact_frame_auto, peak_ws = detect_contact_from_wrist_speed(wrist_speed_series, fps)

    # Smoothness
    elbow_vals, elbow_deltas, elbow_var = compute_smoothness(elbow_list)
    spine_vals, spine_deltas, spine_var = compute_smoothness(spine_list)

    # Static plot
    # plot_smoothness(time_list, elbow_vals, spine_vals, SMOOTHNESS_PNG_PATH)

    # Also save a final static again (explicit filename expected sometimes)
    plt.figure(figsize=(10,4))
    plt.plot(time_list, elbow_vals, label="Elbow angle (deg)")
    plt.plot(time_list, spine_vals, label="Spine lean (deg)")
    plt.xlabel("Time (s)"); plt.ylabel("Angle (deg)")
    plt.title("Temporal Trends: Elbow & Spine")
    plt.legend(loc="best"); plt.tight_layout()
    plt.savefig(FINAL_GRAPH_PNG_PATH); plt.close()

    # Animated graph video (separate from annotated)
    # if time_list:
    #     chart_frames = int(2 * fps)  # 2s animation
    #     width, height = int(width), int(height)
    #     # graph_writer = cv2.VideoWriter(SMOOTH_GRAPH_VIDEO, cv2.VideoWriter_fourcc(*"avc1"), fps, (width, height))
    #     for i in range(chart_frames):
    #         progress = max(1, int(len(time_list) * (i / chart_frames)))
    #         plt.figure(figsize=(10,4))
    #         plt.plot(time_list[:progress], elbow_vals[:progress], label="Elbow angle (deg)")
    #         plt.plot(time_list[:progress], spine_vals[:progress], label="Spine lean (deg)")
    #         plt.xlabel("Time (s)"); plt.ylabel("Angle (deg)")
    #         plt.title("Temporal Trends: Elbow & Spine")
    #         plt.legend(loc="best"); plt.tight_layout()
    #         plt.savefig("temp_chart.png"); plt.close()

    #         chart_img = cv2.imread("temp_chart.png")
    #         if chart_img is None:
    #             continue
    #         chart_img = cv2.resize(chart_img, (width, height))
    #         # put avg fps onto chart video
    #         cv2.putText(chart_img, f"Avg FPS: {avg_fps:.1f}", (width-220, 36),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3)
    #         cv2.putText(chart_img, f"Avg FPS: {avg_fps:.1f}", (width-220, 36),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    #         graph_writer.write(chart_img)
    #     graph_writer.release()
    #     if os.path.exists("temp_chart.png"):
    #         os.remove("temp_chart.png")

    # Bat path straightness & impact bat angle
    straightness = path_straightness(bat_angles)
    bat_angle_at_impact = None
    if impact_frame_auto is not None and 0 <= impact_frame_auto < len(bat_angles):
        bat_angle_at_impact = bat_angles[impact_frame_auto]

    # Per-frame CSV
    df = pd.DataFrame(metrics)
    df.to_csv(PER_FRAME_CSV_PATH, index=False)

    # Averages for scoring
    def avg_nonan(vals):
        arr = np.array([v for v in vals if v is not None], float)
        return float(arr.mean()) if arr.size else None

    avg_elbow = avg_nonan([m["elbow_angle"] for m in metrics])
    avg_spine = avg_nonan([m["spine_lean"] for m in metrics])
    avg_hk    = avg_nonan([m["head_knee_diff"] for m in metrics])
    avg_foot  = avg_nonan([m["foot_direction_angle"] for m in metrics])

    scores = {
        "Footwork":       score_from_targets(avg_foot,  *DEFAULT_TARGETS["foot_direction_deg"]),
        "Head Position":  score_from_targets(avg_hk,    *DEFAULT_TARGETS["head_knee_diff_px"]),
        "Swing Control":  score_from_targets(avg_elbow, *DEFAULT_TARGETS["elbow_angle_deg"]),
        "Balance":        score_from_targets(avg_spine, *DEFAULT_TARGETS["spine_lean_deg"]),
        "Follow-through": 8
    }
    grade = grade_from_scores(scores)

    evaluation = {
        "video": os.path.basename(video_path),
        "avg_fps": round(avg_fps, 2),
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
            "bat_angle_deg_at_impact": bat_angle_at_impact,
            "bat_path_straightness_0to1": straightness
        },
        "scores": scores,
        "skill_grade": grade,
        "artifacts": {
            "annotated_video": ANNOTATED_VIDEO_PATH,
            "smoothness_final_png": FINAL_GRAPH_PNG_PATH,
            "per_frame_csv": PER_FRAME_CSV_PATH,
            "report_pdf": PDF_REPORT_PATH
        }
    }

    # Persist evaluation(s) to JSON (append)
    if os.path.exists(EVAL_JSON_PATH):
        try:
            data = json.load(open(EVAL_JSON_PATH, "r"))
            if not isinstance(data, list):
                data = [data]
        except Exception:
            data = []
    else:
        data = []
    data.append(evaluation)
    with open(EVAL_JSON_PATH, "w") as f:
        json.dump(data, f, indent=4)

    # PDF report
    make_pdf_report(PDF_REPORT_PATH, evaluation, FINAL_GRAPH_PNG_PATH, ANNOTATED_VIDEO_PATH)

    return evaluation

# -----------------------------
# CLI entry
# -----------------------------
if __name__ == "__main__":
    VIDEO_PATH = "cricket.mp4"  # change to your input
    out = analyze_video(VIDEO_PATH)
    print(json.dumps(out, indent=4))
    print("\nArtifacts:")
    for k, v in out["artifacts"].items():
        print(f" - {k}: {v}")
