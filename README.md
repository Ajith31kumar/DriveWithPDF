AthleteRise - AI-Powered Cover Drive Analyzer
This project is a real-time cricket video analysis system that processes a cover drive shot, performs per-frame pose estimation, and provides a final evaluation with actionable feedback. The system is built to be a modular and reproducible tool for biomechanical analysis.

🚀 Setup & Run Instructions
To set up and run the application, follow these steps:

Clone the repository:

git clone [your-repo-url]
cd [your-repo-name]

Install dependencies:
It is recommended to use a virtual environment. Install all required packages using the requirements.txt file.

pip install -r requirements.txt

Run the Streamlit application:
Launch the app from your terminal using the following command:

streamlit run cover_drive_analysis_realtime.py

This will open a web browser window displaying the application.

Use the app:

Upload a video file of a cricket cover drive using the file uploader.

Adjust the biomechanical thresholds in the sidebar if you wish to customize the analysis.

Click the "🚀 Process Video" button.

The app will process the video, display the annotated video with overlays, and show the final evaluation summary. You can then download the video and the evaluation report.

📝 Notes on Assumptions & Limitations
Pose Estimation Model: This project utilizes a lightweight pose estimation model to achieve the real-time performance target (
ge10 FPS). This may result in some inaccuracies with occluded or fast-moving joints.

Video Format: The system is optimized for standard video formats like .mp4, .mov, and .avi.

Reference Comparison: The targets.json file contains default biomechanical thresholds that are used as a baseline for an "ideal" cover drive. These can be adjusted by the user in the Streamlit app's sidebar.

Bat Detection: The base scope does not include advanced bat tracking. Bat-related metrics are approximate and based on lightweight heuristics.
