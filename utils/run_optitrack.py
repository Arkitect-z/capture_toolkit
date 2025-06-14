import os
import sys

root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from optitrack_processor import OptitrackConfiguration, OptiTrackProcessor

if __name__ == "__main__":
    config = OptitrackConfiguration(
        file_path=os.path.join("data_pilot", "OptiTrack", "Take 2025-06-13 02.55.37 PM.csv"),
        model_type="smpl",
        gender="neutral",
        batch_size=128,
        output_folder="./optitrack_export",
        video_filename="human_motion.mp4",
        motion_filename="smpl_motion.npz",
        video_fps=120,
        image_render_size=1024
    )

    try:
        processor = OptiTrackProcessor(config)
        
        print("\n--- Starting Motion Data Export ---")
        processor.export_motion_data()
        
        print("\n--- Starting Video Export ---")
        processor.export_video()

    except (ValueError, FileNotFoundError) as e:
        print(f"An error occurred: {e}")
