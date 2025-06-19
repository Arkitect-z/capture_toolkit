import os
import glob
import traceback

from tactile_processor import TactileConfiguration, TactileProcessor

def main():
    """
    Main function to configure and run the batch processing of tactile .csv files,
    with a final summary report of any errors.
    """
    # --- Configuration ---
    input_base_directory = "data_pilot/Tactile"
    output_base_directory = "tactile_export"
    
    # Use glob to find all left and right hand .CSV files
    # We will process Tactile_L and Tactile_R folders separately
    csv_files = glob.glob(os.path.join(input_base_directory, 'Tactile_*', '*.CSV'), recursive=True)
    
    if not csv_files:
        print(f"Error: No .CSV files found in '{input_base_directory}'")
        return

    total_files = len(csv_files)
    print(f"Found {total_files} .CSV files to process.")
    
    # --- Initialization ---
    failed_files = []
    
    configs = []
    for csv_file in csv_files:
        relative_path = os.path.relpath(csv_file, input_base_directory)
        file_basename_without_ext = os.path.splitext(os.path.basename(relative_path))[0]
        hand_type_folder = os.path.dirname(relative_path) # Tactile_L or Tactile_R
        
        output_folder_for_file = os.path.join(output_base_directory, hand_type_folder, file_basename_without_ext)

        config = TactileConfiguration(
            input_path=csv_file,
            output_folder=output_folder_for_file,
            output_filename="tactile_120hz.csv",
            original_fps=5, # Assuming original data is 5Hz
            target_fps=120   # Target upsampling frequency
        )
        configs.append(config)

    # --- Run processing serially with progress indicators ---
    for i, config in enumerate(configs):
        current_file_num = i + 1
        file_basename = os.path.basename(config.input_path)
        
        print("\n" + "="*80)
        print(f"--- Processing file ({current_file_num}/{total_files}): {file_basename} ---")
        print("="*80)
        
        try:
            processor = TactileProcessor(config)
            processor.process_and_save()
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
    main()
