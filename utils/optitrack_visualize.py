import sys
import os
import torch # Added for device management and tensor operations
import numpy as np # Added for faces conversion if needed
from tqdm import tqdm # For progress bar

# Ensure the root directory is in the Python path to find utils
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from optitrack.utils.csv_utils import read_optitrack_data
from optitrack.utils.smplx_utils import get_smplx_model, get_model_out
# from utils.pytorch3d_utils import single_visualize # No longer used from here
from optitrack.utils.pytorch3d_utils import sequence_visualize # Import the new method
from optitrack.utils.kp_convert import get_body_pose

if __name__ == "__main__":
    # --- Configuration ---
    file_path = os.path.join("data_pilot/OptiTrack", "Take 2025-05-28 12.04.10 AM.csv")
    model_type = "smpl"
    gender = "neutral"
    batch_size = 128 # Number of frames to process in one go by get_model_out
    
    output_folder = "./optitrack_export"
    video_filename = "human_motion.mp4" # Potentially new name for the direct video
    output_video_path = os.path.join(output_folder, video_filename)
    
    video_fps = 120  # Frames per second for the output video
    image_render_size = 1024 # Size of the rendered images/video frames

    # --- Setup Directories ---
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    # --- Device Setup ---
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device) # Set default CUDA device
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # --- Load Data and Model ---
    print(f"Reading OptiTrack data from: {file_path}")
    df_bone, _, human_list, _ = read_optitrack_data(file_path) #
    if not human_list:
        print("Error: No human found in the OptiTrack data.")
        sys.exit(1)
    human_name = human_list[0]
    print(f"Processing data for human: {human_name}")
    print(f"Total frames to process from CSV: {df_bone.shape[0]}")

    print(f"Loading {model_type} model with gender: {gender}")
    smpl_model_tuple = get_smplx_model(model_type=model_type, gender=gender) #
    smpl_model_object = smpl_model_tuple[0] 
    # Prepare faces tensor once and ensure it's on the correct device
    faces_np = smpl_model_object.faces.astype(np.int32)
    faces_tensor = torch.tensor(faces_np, dtype=torch.long).to(device)


    print("Preparing body pose data in batches...")
    # get_body_pose returns a list of dictionaries; each dictionary is a batch of pose data
    batched_input_args = get_body_pose(
        df_bone, model_type=model_type, human_name=human_name, batch=batch_size
    ) #
    print(f"Data prepared into {len(batched_input_args)} batches.")

    # --- Accumulate All Frame Vertices ---
    all_frames_vertices = [] # List to store vertices [num_verts, 3] for each frame
    
    print("Processing batches to extract all frame vertices...")
    # Use tqdm for a progress bar over the batches
    for current_batch_input_args in tqdm(batched_input_args, desc="Extracting Vertices from Batches"):
        # Ensure all tensors in the current batch_input_args are on the 'device'
        for key in current_batch_input_args:
            if isinstance(current_batch_input_args[key], torch.Tensor):
                current_batch_input_args[key] = current_batch_input_args[key].to(device)
        
        # model_tuple is (model_object, model_type_string), both should be on device from get_smplx_model
        output_for_batch = get_model_out(model=smpl_model_tuple, input_args=current_batch_input_args) #
        
        # output_for_batch.vertices is a tensor of shape (frames_in_this_batch, num_vertices, 3)
        num_frames_in_current_batch = output_for_batch.vertices.shape[0]

        for frame_idx_in_batch in range(num_frames_in_current_batch):
            # Detach from computation graph.
            # Vertices are kept on the original device; sequence_visualize will handle its device needs.
            all_frames_vertices.append(output_for_batch.vertices[frame_idx_in_batch].detach())

    total_collected_frames = len(all_frames_vertices)
    print(f"Total {total_collected_frames} frames' vertices collected.")
    if total_collected_frames == 0 and df_bone.shape[0] > 0:
        print("Warning: No vertices were collected, but CSV data exists. Check batch processing and model output.")


    # --- Generate Video Directly Using the New Method ---
    if all_frames_vertices:
        print(f"Calling sequence_visualize to generate video at {output_video_path}...")
        sequence_visualize(
            all_vertices_list=all_frames_vertices,
            faces_tensor=faces_tensor, # Pass the faces tensor
            output_video_path=output_video_path,
            fps=video_fps,
            image_size=image_render_size, 
            device=device # Pass the primary device being used
        )
    else:
        print("No vertices were processed or collected. Video creation skipped.")

    print("Processing complete.")