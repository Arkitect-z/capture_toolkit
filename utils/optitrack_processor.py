import os
import torch
import numpy as np
from tqdm import tqdm

# It's good practice for a library to handle its own dependencies.
# Assuming these utils are part of your 'optitrack' package.
from optitrack.utils.csv_utils import read_optitrack_data, get_joint_position
from optitrack.utils.smplx_utils import get_smplx_model, get_model_out
from optitrack.utils.kp_convert import get_body_pose
from optitrack.utils.pytorch3d_utils import sequence_visualize

class OptitrackConfiguration:
    """
    A class to hold all configuration parameters for OptiTrack data processing.
    This makes it easy to pass settings to the processing function.
    """
    def __init__(self,
                 file_path="data_pilot/OptiTrack/Take 2025-06-13 02.55.37 PM.csv",
                 model_type="smplh",
                 gender="neutral",
                 batch_size=128,
                 output_folder="./optitrack_export",
                 video_filename="human_motion.mp4",
                 motion_filename="smpl_motion.npz",
                 video_fps=120,
                 image_render_size=1024):
        self.file_path = file_path
        self.model_type = model_type
        self.gender = gender
        self.batch_size = batch_size
        self.output_folder = output_folder
        self.video_filename = video_filename
        self.motion_filename = motion_filename
        self.video_fps = video_fps
        self.image_render_size = image_render_size
        
        # Derived paths
        self.output_video_path = os.path.join(self.output_folder, self.video_filename)
        self.output_motion_path = os.path.join(self.output_folder, self.motion_filename)

class OptiTrackProcessor:
    """
    A processor class to handle OptiTrack data conversion.
    It separates data loading, motion export, and video rendering into distinct methods.
    """
    def __init__(self, config: OptitrackConfiguration):
        """
        Initializes the processor, sets up the environment, and loads data.
        """
        self.config = config
        self.device = None
        self.df_bone = None
        self.human_name = None
        self.smpl_model_tuple = None
        self.faces_tensor = None
        self._batched_input_args = None
        self._all_frames_vertices = None

        self._setup_environment()
        self._load_data_and_model()

    def _setup_environment(self):
        """Sets up the output directory and PyTorch device."""
        if not os.path.exists(self.config.output_folder):
            os.makedirs(self.config.output_folder)
            print(f"Created output folder: {self.config.output_folder}")

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(self.device)
        print(f"Using device: {self.device}")

    def _load_data_and_model(self):
        """Loads OptiTrack CSV data and the SMPL model."""
        print(f"Reading OptiTrack data from: {self.config.file_path}")
        df_bone, _, human_list, _ = read_optitrack_data(self.config.file_path)
        if not human_list:
            raise ValueError("Error: No human found in the OptiTrack data.")

        self.df_bone = df_bone
        self.human_name = human_list[0]
        print(f"Processing data for human: {self.human_name}")
        print(f"Total frames to process from CSV: {self.df_bone.shape[0]}")

        print(f"Loading {self.config.model_type} model with gender: {self.config.gender}")
        self.smpl_model_tuple = get_smplx_model(model_type=self.config.model_type, gender=self.config.gender)
        smpl_model_object = self.smpl_model_tuple[0]
        faces_np = smpl_model_object.faces.astype(np.int32)
        self.faces_tensor = torch.tensor(faces_np, dtype=torch.long).to(self.device)

    def _get_pose_data(self):
        """Internal method to process and cache body pose data."""
        if self._batched_input_args is None:
            print("Preparing body pose data in batches...")
            self._batched_input_args = get_body_pose(
                self.df_bone, model_type=self.config.model_type, human_name=self.human_name, batch=self.config.batch_size
            )
            print(f"Data prepared into {len(self._batched_input_args)} batches.")
        return self._batched_input_args
        
    def _get_vertices(self):
        """Internal method to calculate and cache all frame vertices."""
        if self._all_frames_vertices is None:
            batched_input_args = self._get_pose_data()
            if not batched_input_args:
                print("Warning: No pose data to generate vertices from.")
                self._all_frames_vertices = []
                return self._all_frames_vertices

            all_verts = []
            print("Processing batches to extract all frame vertices...")
            for current_batch in tqdm(batched_input_args, desc="Extracting Vertices"):
                for key in current_batch:
                    if isinstance(current_batch[key], torch.Tensor):
                        current_batch[key] = current_batch[key].to(self.device)
                
                output_for_batch = get_model_out(model=self.smpl_model_tuple, input_args=current_batch)
                
                for frame_idx in range(output_for_batch.vertices.shape[0]):
                    all_verts.append(output_for_batch.vertices[frame_idx].detach())
            
            self._all_frames_vertices = all_verts
            print(f"Total {len(self._all_frames_vertices)} frames' vertices collected.")
        
        return self._all_frames_vertices

    def export_motion_data(self):
        """
        Generates and saves the SMPL motion data in an .npz file (AMASS format).
        """
        batched_input_args = self._get_pose_data()
        if not batched_input_args:
            print("No pose data was generated, skipping SMPL motion data export.")
            return

        print("Aggregating and saving data in AMASS format...")
        full_global_orient = torch.cat([b['global_orient'] for b in batched_input_args], dim=0)
        full_body_pose = torch.cat([b['body_pose'] for b in batched_input_args], dim=0)
        poses = torch.cat([full_global_orient, full_body_pose], dim=1).cpu().numpy()
        trans = get_joint_position(self.df_bone, "Hip", self.human_name).cpu().numpy()
        betas_num = 10 if self.config.model_type == 'smpl' else 16
        betas = np.zeros(betas_num)
        dmpls = np.zeros((poses.shape[0], 8))

        np.savez(
            self.config.output_motion_path,
            poses=poses, trans=trans, betas=betas,
            mocap_framerate=self.config.video_fps, gender=str(self.config.gender), dmpls=dmpls
        )
        print(f"SMPL motion data saved successfully to {self.config.output_motion_path}")

    def export_video(self):
        """
        Generates a video visualization from the motion data.
        """
        all_frames_vertices = self._get_vertices()
        
        if not all_frames_vertices:
            print("No vertices were collected. Video creation skipped.")
            return

        print(f"Generating video at {self.config.output_video_path}...")
        sequence_visualize(
            all_vertices_list=all_frames_vertices,
            faces_tensor=self.faces_tensor,
            output_video_path=self.config.output_video_path,
            fps=self.config.video_fps,
            image_size=self.config.image_render_size,
            device=self.device
        )
