import pyrealsense2 as rs
import numpy as np
import cv2
import os
import shutil
import datetime
from tqdm import tqdm

class RealSenseConfiguration:
    """
    A class to hold all configuration parameters for RealSense data processing.
    This makes it easy to pass settings to the processing function and 
    control which data streams are exported.
    """
    def __init__(self,
                 bag_file_path: str,
                 output_base_dir: str = "realsense_export",
                 # --- Export controls ---
                 export_color_frames: bool = False,
                 export_depth_frames_raw: bool = False,
                 export_depth_frames_visualized: bool = False,
                 export_rgb_video: bool = True,
                 export_depth_video: bool = True,
                 export_imu_data: bool = True,
                 export_specifications: bool = True,
                 # --- Video settings ---
                 video_codec: str = 'mp4v',
                 depth_colormap_enabled: bool = True):
        
        if not os.path.exists(bag_file_path):
            raise FileNotFoundError(f"Bag file not found at '{bag_file_path}'")

        self.bag_file_path = bag_file_path
        # Each bag file will have its own output subdirectory to prevent overwriting.
        self.output_base_dir = os.path.join(output_base_dir, os.path.splitext(os.path.basename(bag_file_path))[0])
        
        # Export controls
        self.export_color_frames = export_color_frames
        self.export_depth_frames_raw = export_depth_frames_raw
        self.export_depth_frames_visualized = export_depth_frames_visualized
        self.export_rgb_video = export_rgb_video
        self.export_depth_video = export_depth_video
        self.export_imu_data = export_imu_data
        self.export_specifications = export_specifications
        
        # Other settings
        self.depth_colormap_enabled = depth_colormap_enabled
        self.video_codec = video_codec

        # --- Derived output paths ---
        self.output_color_frames_dir = os.path.join(self.output_base_dir, "frames", "color")
        self.output_depth_raw_dir = os.path.join(self.output_base_dir, "frames", "depth_raw_npy")
        self.output_depth_viz_dir = os.path.join(self.output_base_dir, "frames", "depth_visualized_png")
        self.output_video_dir = os.path.join(self.output_base_dir, "videos")
        self.output_imu_dir = os.path.join(self.output_base_dir, "imu_data")
        self.output_spec_dir = os.path.join(self.output_base_dir, "specifications")


class RealSenseProcessor:
    """
    A class for processing Intel RealSense .bag files.
    It can extract, align, and save RGB and depth image data, IMU data, 
    and stream specification details into separate, modular functions.
    """
    def __init__(self, config: RealSenseConfiguration):
        """
        Initializes the RealSenseBagProcessor.
        Args:
            config (RealSenseConfiguration): Configuration object with all settings.
        """
        self.config = config
        self.pipeline = rs.pipeline()
        self.rs_config = rs.config()
        
        rs.config.enable_device_from_file(self.rs_config, self.config.bag_file_path, repeat_playback=False)
        
        self.align = rs.align(rs.stream.color)

        # Stream profiles and properties
        self.profile = None
        self.video_width = 0
        self.video_height = 0
        self.video_fps = 30.0

        self._initialize_stream_profiles()

    def _setup_environment(self):
        """Creates all necessary output directories based on configuration."""
        dirs_to_create = []
        if self.config.export_rgb_video or self.config.export_depth_video:
            dirs_to_create.append(self.config.output_video_dir)
        if self.config.export_imu_data:
            dirs_to_create.append(self.config.output_imu_dir)
        if self.config.export_specifications:
            dirs_to_create.append(self.config.output_spec_dir)
        if self.config.export_color_frames:
            dirs_to_create.append(self.config.output_color_frames_dir)
        if self.config.export_depth_frames_raw:
            dirs_to_create.append(self.config.output_depth_raw_dir)
        if self.config.export_depth_frames_visualized:
            dirs_to_create.append(self.config.output_depth_viz_dir)
            
        for d in dirs_to_create:
            os.makedirs(d, exist_ok=True)
        # No print statement here to keep the main script's output cleaner.

    def _initialize_stream_profiles(self):
        """Initializes the pipeline to obtain stream information."""
        try:
            self.profile = self.rs_config.resolve(self.pipeline)
            color_profile = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
            self.video_width = color_profile.width()
            self.video_height = color_profile.height()
            self.video_fps = float(color_profile.fps())
        except Exception as e:
            raise RuntimeError(f"Could not initialize stream profiles from .bag file: {e}")

    def _write_intrinsics(self, file, profile):
        """Helper to write stream intrinsics to a file."""
        intrinsics = profile.get_intrinsics()
        file.write(f"    - fx, fy (focal length): {intrinsics.fx}, {intrinsics.fy}\n")
        file.write(f"    - ppx, ppy (principal point): {intrinsics.ppx}, {intrinsics.ppy}\n")
        file.write(f"    - width, height: {intrinsics.width}, {intrinsics.height}\n")
        file.write(f"    - Distortion Model: {intrinsics.model}\n")
        file.write(f"    - Distortion Coeffs: {intrinsics.coeffs}\n")

    def export_specifications(self):
        """Saves detailed specifications of the camera streams to a text file."""
        spec_file_path = os.path.join(self.config.output_spec_dir, "stream_specifications.txt")
        with open(spec_file_path, 'w') as f:
            f.write(f"Specifications for: {os.path.basename(self.config.bag_file_path)}\n")
            f.write(f"Processed on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            color_stream = self.profile.get_stream(rs.stream.color)
            if color_stream:
                f.write("--- RGB Color Stream ---\n")
                self._write_intrinsics(f, color_stream.as_video_stream_profile())

            depth_stream = self.profile.get_stream(rs.stream.depth)
            if depth_stream:
                f.write("\n--- Depth Stream ---\n")
                depth_sensor = self.profile.get_device().first_depth_sensor()
                f.write(f"- Depth Scale: {depth_sensor.get_depth_scale():.6f}\n")
                self._write_intrinsics(f, depth_stream.as_video_stream_profile())

    def run_full_export(self):
        """
        Processes the entire .bag file at once and saves all configured outputs.
        This is the most efficient method as it only iterates through the file once.
        """
        self._setup_environment()
        
        if self.config.export_specifications:
            self.export_specifications()

        # --- Initialize writers ---
        color_writer, depth_writer, imu_file = None, None, None
        
        if self.config.export_rgb_video:
            ext = 'mp4' if 'mp4' in self.config.video_codec else 'avi'
            path = os.path.join(self.config.output_video_dir, f"rgb_video.{ext}")
            fourcc = cv2.VideoWriter_fourcc(*self.config.video_codec)
            color_writer = cv2.VideoWriter(path, fourcc, self.video_fps, (self.video_width, self.video_height))

        if self.config.export_depth_video:
            ext = 'mp4' if 'mp4' in self.config.video_codec else 'avi'
            path = os.path.join(self.config.output_video_dir, f"depth_visualized_video.{ext}")
            fourcc = cv2.VideoWriter_fourcc(*self.config.video_codec)
            depth_writer = cv2.VideoWriter(path, fourcc, self.video_fps, (self.video_width, self.video_height))

        if self.config.export_imu_data:
            path = os.path.join(self.config.output_imu_dir, "imu_data.csv")
            imu_file = open(path, 'w')
            imu_file.write("timestamp_ms,stream_type,x,y,z\n")

        # --- Processing Loop ---
        pbar = None
        try:
            self.profile = self.pipeline.start(self.rs_config)
            playback = self.profile.get_device().as_playback()
            playback.set_real_time(False)
            
            total_frames = int(playback.get_duration().total_seconds() * self.video_fps)
            pbar = tqdm(total=total_frames, desc=f"Exporting data", unit="frame", leave=False)

            frame_count = 0
            while True:
                try:
                    frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                except RuntimeError:
                    break
                
                if imu_file:
                    for frame in frames:
                        if frame.is_motion_frame():
                            motion = frame.as_motion_frame()
                            stream_type = "Accel" if motion.get_profile().stream_type() == rs.stream.accel else "Gyro"
                            imu_file.write(f"{motion.get_timestamp():.6f},{stream_type},{motion.get_motion_data().x:.6f},{motion.get_motion_data().y:.6f},{motion.get_motion_data().z:.6f}\n")

                aligned_frames = self.align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()

                if color_frame and depth_frame:
                    pbar.update(1)
                    timestamp_ms = int(color_frame.get_timestamp())
                    color_image = cv2.cvtColor(np.asanyarray(color_frame.get_data()), cv2.COLOR_RGB2BGR)
                    depth_raw = np.asanyarray(depth_frame.get_data())
                    
                    if self.config.export_color_frames:
                        cv2.imwrite(os.path.join(self.config.output_color_frames_dir, f"color_{timestamp_ms:013d}.png"), color_image)
                    if self.config.export_depth_frames_raw:
                        np.save(os.path.join(self.config.output_depth_raw_dir, f"depth_raw_{timestamp_ms:013d}.npy"), depth_raw)
                    if color_writer:
                        color_writer.write(color_image)
                    if depth_writer or self.config.export_depth_frames_visualized:
                        depth_viz = cv2.applyColorMap(cv2.convertScaleAbs(depth_raw, alpha=0.03), cv2.COLORMAP_JET)
                        if self.config.export_depth_frames_visualized:
                            cv2.imwrite(os.path.join(self.config.output_depth_viz_dir, f"depth_viz_{timestamp_ms:013d}.png"), depth_viz)
                        if depth_writer:
                            depth_writer.write(depth_viz)
                    frame_count += 1
        finally:
            if pbar: pbar.close()
            if color_writer: color_writer.release()
            if depth_writer: depth_writer.release()
            if imu_file: imu_file.close()
            self.pipeline.stop()
            print(f"Export finished. {frame_count} frame pairs processed.")
            print(f"Outputs saved in: {os.path.abspath(self.config.output_base_dir)}")