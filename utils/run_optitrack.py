import os
import glob
import sys
import traceback

# Ensure the utility modules can be found by adding the script's directory to the path
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from optitrack_processor import OptitrackConfiguration, OptiTrackProcessor

def process_optitrack_data(configs):
    """
    Main function to run the BATCH/SERIAL processing of OptiTrack .csv files based on provided configurations.
    Includes a final summary report of any errors.

    Args:
        configs (list[OptitrackConfiguration]): A list of configuration objects to process.
    """
    # --- Initialization ---
    failed_files = []
    total_files = len(configs)

    # --- Run processing serially with progress indicators ---
    for i, config in enumerate(configs):
        current_file_num = i + 1
        file_basename = os.path.basename(config.file_path)
        
        print("\n" + "="*80)
        print(f"--- Processing file ({current_file_num}/{total_files}): {file_basename} ---")
        print("="*80)
        
        try:
            processor = OptiTrackProcessor(config)
            
            print(f"Starting Motion Data Export for {file_basename}...")
            processor.export_motion_data()
            
            print(f"Starting Video Export for {file_basename}...")
            processor.export_video()
            
            print(f"\n[SUCCESS] Finished processing {file_basename}")

        except Exception as e:
            # If an error occurs, print it and add the file to the list of failures
            print(f"\n[ERROR] An unexpected error occurred with {file_basename}: {e}\n{traceback.format_exc()}")
            failed_files.append(file_basename)

    # --- Final Summary Report ---
    print("\n" + "="*80)
    print("--- Final Processing Summary ---")
    if not failed_files:
        print("✅ All files were processed successfully!")
    else:
        print(f"❌ Encountered errors in {len(failed_files)} out of {total_files} files:")
        for filename in failed_files:
            print(f"  - {filename}")
    print("="*80)
    
    print("\n--- All serial processing tasks are complete. ---")

if __name__ == "__main__":
    """
    This block is executed when the script is run directly.
    It discovers .csv files, creates configurations, and calls the main processing function.
    """
    # --- Configuration ---
    input_directory = "data_pilot/OptiTrack"
    output_base_directory = "optitrack_export"
    
    # Use glob to find all .csv files in the input directory
    csv_files = glob.glob(os.path.join(input_directory, "*M.csv"))
    if not csv_files:
        print(f"Error: No .csv files found in '{input_directory}'")
    else:
        print(f"Found {len(csv_files)} .csv files to process serially.")
        
        configurations = []
        for csv_file in csv_files:
            file_basename = os.path.splitext(os.path.basename(csv_file))[0]
            output_folder_for_file = os.path.join(output_base_directory, file_basename)

            config = OptitrackConfiguration(
                file_path=csv_file,
                output_folder=output_folder_for_file,
                model_type="smplx",
                gender="neutral",
                batch_size=128,
                video_filename="human_motion.mp4",
                motion_filename="smplx_motion.npz",
                video_fps=120,
                image_render_size=1024
            )
            configurations.append(config)

        # Call the main function with the generated configurations
        process_optitrack_data(configurations)
