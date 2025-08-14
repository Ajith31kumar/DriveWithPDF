import os
import io
import json
import time
import tempfile
from pathlib import Path

import streamlit as st
import base64    
# --- import your backend ---
# If your file is named differently, change 'backend' to your module name.
try:
    from cover_drive_analysis_realtime import (
        analyze_video,
        OUTPUT_DIR,
        ANNOTATED_VIDEO_PATH,
        TARGETS_JSON_PATH,
        EVAL_JSON_PATH,
        PDF_REPORT_PATH,
    )
except Exception as e:
    st.error("Could not import backend. Make sure backend.py is in the same folder.")
    st.exception(e)
    st.stop()

st.set_page_config(page_title="AthleteRise – Cover Drive Analyzer", layout="wide")

# ----------------------------
# Helpers
# ----------------------------
def load_targets(path: str):
    # targets.json is created by backend.ensure_targets_config() on first run,
    # but we may want to pre-load if present so user can edit
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

def save_targets(path: str, targets_dict: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(targets_dict, f, indent=4)

def read_file_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()

def most_recent_eval(eval_path: str):
    if not os.path.exists(eval_path):
        return None
    try:
        data = json.load(open(eval_path, "r"))
        if isinstance(data, list) and data:
            return data[-1]
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None
targets = load_targets(TARGETS_JSON_PATH)

def find_optional_artifacts(output_dir: str):
    """Look for graph-only video and final graph image saved by bonus block."""
    graph_mp4 = None
    graph_png = None
    # Common names used in our backend variants
    for name in ["smoothness_graph.mp4", "annotated_bonus_slate.mp4"]:
        p = os.path.join(output_dir, name)
        if os.path.exists(p):
            graph_mp4 = p
            break
    for name in ["smoothness_final.png", "smoothness.png"]:
        p = os.path.join(output_dir, name)
        if os.path.exists(p):
            graph_png = p
            break
    return graph_mp4, graph_png
if 'rerun_triggered' not in st.session_state:
    st.session_state.rerun_triggered = False

# --- Callback function to trigger the "rerun" action ---
def rerun_section_action():
    """Sets a state variable to True, causing the conditional block to execute."""
    st.session_state.rerun_triggered = True

# --- The "re-runnable" code logic ---
def do_something_after_save():
    """This function contains the code you want to run after the button is clicked."""
    st.subheader("Action Triggered!")
    st.write("This part of the code was 're-run' because the button was clicked.")
    st.info(f"The current saved targets are: {load_targets(TARGETS_JSON_PATH)}")
    # Reset the state so the action doesn't re-run on every future interaction
    st.session_state.rerun_triggered = False
def report_save():
    if os.path.exists(PDF_REPORT_PATH):
        st.download_button(
            "Download Report file",
            data=read_file_bytes(PDF_REPORT_PATH),
            file_name=Path(PDF_REPORT_PATH).name,
            mime="application/pdf",
        )
    else:
        st.sidebar.info("No targets.json yet. It will be created on first run by the backend.")
# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("⚙️ Controls")

# Load current targets to edit
# targets = load_targets(TARGETS_JSON_PATH)
if targets:
    st.sidebar.markdown("**Thresholds (edit & save)**")
    # Render number inputs for each range pair
    edited_targets = {}
    for key, val in targets.items():
        if isinstance(val, (list, tuple)) and len(val) == 2:
            lo, hi = val
            c1, c2 = st.sidebar.columns(2)
            with c1:
                new_lo = st.number_input(f"{key} (lo)", value=float(lo), step=1.0, key=f"{key}_lo")
            with c2:
                new_hi = st.number_input(f"{key} (hi)", value=float(hi), step=1.0, key=f"{key}_hi")
            # keep as int when clean, else float
            if float(new_lo).is_integer() and float(new_hi).is_integer():
                edited_targets[key] = [int(new_lo), int(new_hi)]
            else:
                edited_targets[key] = [float(new_lo), float(new_hi)]
        else:
            # Just echo values we don’t recognize as ranges
            edited_targets[key] = val

    if st.sidebar.button("💾 Save thresholds"):
        save_targets(TARGETS_JSON_PATH, edited_targets)
        st.sidebar.success("Saved thresholds to targets.json")
        

else:
    st.sidebar.info("No targets.json yet. It will be created on first run by the backend.")
    st.sidebar.button("Re-run this part", on_click=rerun_section_action)

st.sidebar.markdown("---")
if os.path.exists(PDF_REPORT_PATH):
    st.sidebar.download_button(
        "Download Report file",
        data=read_file_bytes(PDF_REPORT_PATH),
        file_name=Path(PDF_REPORT_PATH).name,
        mime="application/pdf",
    )
st.sidebar.markdown("---")
st.sidebar.caption("Tip: Larger videos may take longer to process.")

# ----------------------------
# Header
# ----------------------------
st.title("🏏 AthleteRise ")
st.write("Upload your batting video, run the analyzer, and download the results.")

# ----------------------------
# Upload
# ----------------------------
uploaded = st.file_uploader("Upload a cricket batting video", type=["mp4", "mov", "mkv", "avi"])
run_btn = st.button("🚀 Process Video")

# ----------------------------
# Main flow
# ----------------------------
if run_btn:
    if not uploaded:
        st.warning("Please upload a video first.")
        st.stop()

    # Save uploaded file to a temp path
    tmpdir = tempfile.mkdtemp(prefix="athleterise_")
    in_path = os.path.join(tmpdir, uploaded.name)
    with open(in_path, "wb") as f:
        f.write(uploaded.getbuffer())

    st.info("Starting analysis…")
    placeholder = st.empty()
    t0 = time.time()
    try:
        # NOTE: analyze_video writes artifacts to OUTPUT_DIR set by backend
        with st.spinner("Running backend analysis…"):
            result = analyze_video(in_path)
        dt = time.time() - t0
        placeholder.success(f"✅ Done in {dt:.1f}s")
    except Exception as e:
        placeholder.error("❌ Backend failed.")
        st.exception(e)
        st.stop()

    # ------------------------
    # Results area
    # ------------------------
    # ANNOTATED_VIDEO_PATH = "cricket.mp4"


    st.subheader("🎥 Annotated Video")

    if os.path.exists(ANNOTATED_VIDEO_PATH):
        # Read and encode video
        video_bytes = read_file_bytes(ANNOTATED_VIDEO_PATH)
        video_base64 = base64.b64encode(video_bytes).decode()

        # Custom video display
        st.markdown(
            f"""
            <video width="800" height="450" controls>
                <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
            </video>
            """,
            unsafe_allow_html=True
        )

        # Download button
        st.download_button(
            "⬇️ Download annotated video",
            data=video_bytes,
            file_name=Path(ANNOTATED_VIDEO_PATH).name,
            mime="video/mp4",
        )
    else:
        st.warning("Annotated video not found.")


    # Optional artifacts from bonus graph feature
    graph_mp4, graph_png = find_optional_artifacts(OUTPUT_DIR)



    st.subheader("🖼️ Final Graph Screenshot")
    # Prefer the explicit screenshot if exists; otherwise show smoothness.png
    img_path = graph_png or os.path.join(OUTPUT_DIR, "smoothness.png")
    if os.path.exists(img_path):
        st.image(img_path, use_container_width=True)
        st.download_button(
            "⬇️ Download graph image",
            data=read_file_bytes(img_path),
            file_name=Path(img_path).name,
            mime="image/png",
        )
    else:
        st.info("No graph image found yet.")

    # Evaluation JSON (this run’s result, plus download full evaluation.json)
    st.subheader("🧾 Evaluation Summary")
    run_eval = result if isinstance(result, dict) else most_recent_eval(EVAL_JSON_PATH)
    if run_eval:
        # Pretty summary cards
        m = run_eval.get("averages", {})
        s = run_eval.get("scores", {})
        grade = run_eval.get("skill_grade", "N/A")
        avg_fps = run_eval.get("avg_fps", None)

        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        kpi1.metric("Skill Grade", grade)
        kpi2.metric("Avg FPS", f"{avg_fps:.1f}" if avg_fps is not None else "-")
        kpi3.metric("Elbow ∠ (avg)", f"{m.get('elbow_angle_deg', '-')}")
        kpi4.metric("Spine Lean (avg)", f"{m.get('spine_lean_deg', '-')}")
        kpi5.metric("Head–Knee Δ (avg)", f"{m.get('head_knee_diff_px', '-')} px")

        st.markdown("**Scores**")
        st.json(s)

        st.markdown("**Raw evaluation (this run)**")
        st.json(run_eval)
    else:
        st.info("No evaluation data was returned.")

    # Download full evaluation.json
    if os.path.exists(EVAL_JSON_PATH):
        st.download_button(
            "⬇️ Download evaluation.json",
            data=read_file_bytes(EVAL_JSON_PATH),
            file_name=Path(EVAL_JSON_PATH).name,
            mime="application/json",
        )
    if os.path.exists(PDF_REPORT_PATH):
        st.download_button(
            "Download Report file",
            data=read_file_bytes(PDF_REPORT_PATH),
            file_name=Path(PDF_REPORT_PATH).name,
            mime="application/pdf",
        )

# --------------------------------
# Footer: show the most recent run if user didn’t click Process
# --------------------------------
else:
    st.caption("Tip: upload a video and click **Process Video**. Below shows your last run (if any).")
    last = most_recent_eval(EVAL_JSON_PATH)
    if last:
        st.markdown("### Last Run Snapshot")
        st.json(last)
        bonus_video_path = os.path.join("output", "annotated_bonus.mp4")
        if os.path.exists(bonus_video_path):
            with open(bonus_video_path, "rb") as f:
                st.video(f.read())
        else:
            st.warning("Annotated video not found.")

