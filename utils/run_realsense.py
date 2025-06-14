import os
from realsense_processor import RealSenseConfiguration, RealSenseProcessor

if __name__ == "__main__":
    # --- Configuration ---
    # Define all processing settings here
    config = RealSenseConfiguration(
        bag_file_path="data_pilot/RealSense/20250528_000356.bag",
        output_base_dir="realsense_export",
        save_individual_frames=True,
        save_video=True,
        save_imu_data=True,
        video_codec='mp4v'  # Use 'mp4v' for .mp4, 'XVID' for .avi
    )

    # Check if the .bag file exists before proceeding
    if not os.path.exists(config.bag_file_path):
        print(f"Error: Bag file not found at '{config.bag_file_path}'")
        print("Please update the 'bag_file_path' in 'run_realsense.py'")
    else:
        try:
            # Initialize the processor with the configuration
            processor = RealSenseProcessor(config)
            
            # Run the main processing function
            print("\n--- Starting RealSense Data Export ---")
            processor.process_and_save_all()

        except (FileNotFoundError, RuntimeError) as e:
            print(f"\nAn error occurred during processing: {e}")
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            import traceback
            traceback.print_exc()