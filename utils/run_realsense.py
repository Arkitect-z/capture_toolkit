import os
import glob
import traceback
from realsense_processor import RealSenseConfiguration, RealSenseProcessor

def process_realsense_data(configs):
    """
    Main function to run the BATCH/SERIAL processing of RealSense .bag files based on provided configurations.
    Includes a final summary report of any errors.

    Args:
        configs (list[RealSenseConfiguration]): A list of configuration objects to process.
    """
    # --- Initialization for error reporting ---
    failed_files = []
    total_files = len(configs)

    # --- Run processing serially with progress indicators ---
    for i, config in enumerate(configs):
        current_file_num = i + 1
        file_basename = os.path.basename(config.bag_file_path)
        
        print("\n" + "="*80)
        print(f"--- Processing file ({current_file_num}/{total_files}): {file_basename} ---")
        print("="*80)
        
        try:
            # Initialize the processor with the specific configuration for this file
            processor = RealSenseProcessor(config)
            
            # Run the main export function
            processor.run_full_export()
            
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
    It discovers .bag files, creates configurations, and calls the main processing function.
    """
    # --- Configuration ---
    input_directory = "data_pilot/RealSense"
    output_base_directory = "realsense_export"
    
    # Use glob to find all .bag files in the input directory
    bag_files = glob.glob(os.path.join(input_directory, "*.bag"))
    if not bag_files:
        print(f"Error: No .bag files found in '{input_directory}'")
    else:
        print(f"Found {len(bag_files)} .bag files to process serially.")

        # --- Create a list of configurations for each file ---
        configurations = []
        for bag_file in bag_files:
            config = RealSenseConfiguration(
                bag_file_path=bag_file,
                output_base_dir=output_base_directory,
                
                # --- Select what you want to export ---
                export_rgb_video=True,
                export_depth_video=True,
                export_imu_data=True,
                export_specifications=True,
                export_color_frames=True,
                export_depth_frames_raw=True,
                export_depth_frames_visualized=True,
                
                # --- Other settings ---
                video_codec='mp4v'  # Use 'mp4v' for .mp4, 'XVID' for .avi
            )
            configurations.append(config)

        # Call the main function with the generated configurations
        process_realsense_data(configurations)
