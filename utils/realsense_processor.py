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
    This makes it easy to pass settings to the processing function.
    """
    def __init__(self,
                 bag_file_path="data_pilot/RealSense/20250528_000356.bag",
                 output_base_dir="realsense_export",
                 save_individual_frames=True,
                 save_video=True,
                 save_imu_data=True,
                 depth_colormap_enabled=True,
                 video_codec='mp4v'):
        self.bag_file_path = bag_file_path
        self.output_base_dir = output_base_dir
        self.save_individual_frames = save_individual_frames
        self.save_video = save_video
        self.save_imu_data = save_imu_data
        self.depth_colormap_enabled = depth_colormap_enabled
        self.video_codec = video_codec

        # Derived output paths
        self.output_color_frames_dir = os.path.join(self.output_base_dir, "frames", "color")
        self.output_depth_raw_frames_dir = os.path.join(self.output_base_dir, "frames", "depth_raw_npy")
        self.output_depth_viz_frames_dir = os.path.join(self.output_base_dir, "frames", "depth_visualized_png")
        self.output_video_dir = os.path.join(self.output_base_dir, "videos")
        self.output_imu_dir = os.path.join(self.output_base_dir, "imu_data")
        self.output_spec_dir = os.path.join(self.output_base_dir, "specifications")


class RealSenseProcessor:
    """
    A class for processing Intel RealSense .bag files.
    It can extract, align, and save RGB and depth image data, IMU data, 
    and stream specification details.
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
        
        self._setup_environment()
        
        rs.config.enable_device_from_file(self.rs_config, self.config.bag_file_path, repeat_playback=False)
        
        self.align_to_stream = rs.stream.color
        self.align = rs.align(self.align_to_stream)

        # Stream profiles and video properties
        self.profile = None
        self.color_stream_profile = None
        self.depth_stream_profile = None
        self.accel_stream_profile = None
        self.gyro_stream_profile = None
        self.video_width = 0
        self.video_height = 0
        self.video_fps = 30.0

        self._initialize_streams_and_get_info()

    def _setup_environment(self):
        """Validates file paths and creates all necessary output directories."""
        if not os.path.exists(self.config.bag_file_path):
            raise FileNotFoundError(f".bag file not found: {self.config.bag_file_path}")

        dirs_to_create = [self.config.output_video_dir, self.config.output_imu_dir, self.config.output_spec_dir]
        if self.config.save_individual_frames:
            # Clear old frame directories to prevent mixed data
            for d in [self.config.output_color_frames_dir, self.config.output_depth_raw_frames_dir, self.config.output_depth_viz_frames_dir]:
                if os.path.exists(d):
                    shutil.rmtree(d)
            dirs_to_create.extend([self.config.output_color_frames_dir, self.config.output_depth_raw_frames_dir, self.config.output_depth_viz_frames_dir])

        for d in dirs_to_create:
            os.makedirs(d, exist_ok=True)
        print(f"All output directories ensured under: {self.config.output_base_dir}")
        

    def _initialize_streams_and_get_info(self):
        """
        Initializes the pipeline to obtain stream information for setup.
        """
        try:
            self.profile = self.rs_config.resolve(self.pipeline)
            if not self.profile:
                raise RuntimeError("Could not resolve pipeline configuration from .bag file.")

            # Get video stream info
            color_profile = self.profile.get_stream(rs.stream.color)
            if color_profile:
                self.color_stream_profile = color_profile.as_video_stream_profile()
                self.video_width = self.color_stream_profile.width()
                self.video_height = self.color_stream_profile.height()
                self.video_fps = float(self.color_stream_profile.fps())
                print(f"Video stream info: {self.video_width}x{self.video_height} @ {self.video_fps:.2f} FPS")
            
            # Get other stream profiles
            self.depth_stream_profile = self.profile.get_stream(rs.stream.depth)
            self.accel_stream_profile = self.profile.get_stream(rs.stream.accel)
            self.gyro_stream_profile = self.profile.get_stream(rs.stream.gyro)

        except Exception as e:
            print(f"Error during stream info initialization: {e}")


    def _write_intrinsics(self, file, profile):
        """Helper to write stream intrinsics to a file."""
        intrinsics = profile.get_intrinsics()
        file.write(f"    - fx, fy (focal length): {intrinsics.fx}, {intrinsics.fy}\n")
        file.write(f"    - ppx, ppy (principal point): {intrinsics.ppx}, {intrinsics.ppy}\n")
        file.write(f"    - width, height: {intrinsics.width}, {intrinsics.height}\n")
        file.write(f"    - Distortion Model: {intrinsics.model}\n")
        file.write(f"    - Distortion Coeffs: {intrinsics.coeffs}\n")

    def _save_stream_specifications(self):
        """Saves stream specifications to a text file."""
        spec_file_path = os.path.join(self.config.output_spec_dir, "stream_specifications.txt")
        print(f"Saving stream specifications to: {spec_file_path}")
        with open(spec_file_path, 'w') as f:
            f.write(f"Specifications for: {os.path.basename(self.config.bag_file_path)}\n")
            f.write(f"Processed on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            if self.color_stream_profile:
                f.write("--- RGB Color Stream ---\n")
                self._write_intrinsics(f, self.color_stream_profile)
            if self.depth_stream_profile:
                f.write("\n--- Depth Stream ---\n")
                depth_sensor = self.profile.get_device().first_depth_sensor()
                f.write(f"- Depth Scale: {depth_sensor.get_depth_scale():.6f}\n")
                self._write_intrinsics(f, self.depth_stream_profile.as_video_stream_profile())
        print("Stream specifications saved.")

    def process_and_save_all(self):
        """
        Processes the .bag file based on the initialized configuration.
        """
        self._save_stream_specifications()

        color_video_writer = None
        depth_video_writer = None
        imu_file = None
        pbar = None
        processed_color_frames_count = 0

        try:
            print(f"Starting processing of .bag file: {self.config.bag_file_path}")
            self.profile = self.pipeline.start(self.rs_config)
            
            playback = self.profile.get_device().as_playback()
            playback.set_real_time(False) # Process as fast as possible
            duration_sec = playback.get_duration().total_seconds()
            total_frames_estimate = int(duration_sec * self.video_fps) if duration_sec > 0 else 0
            pbar = tqdm(total=total_frames_estimate, desc="Processing frames", unit="frame")

            # Initialize VideoWriters
            if self.config.save_video and self.color_stream_profile:
                ext = 'mp4' if 'mp4' in self.config.video_codec else 'avi'
                color_path = os.path.join(self.config.output_video_dir, f"rgb_video.{ext}")
                depth_path = os.path.join(self.config.output_video_dir, f"depth_visualized_video.{ext}")
                fourcc = cv2.VideoWriter_fourcc(*self.config.video_codec)
                color_video_writer = cv2.VideoWriter(color_path, fourcc, self.video_fps, (self.video_width, self.video_height))
                depth_video_writer = cv2.VideoWriter(depth_path, fourcc, self.video_fps, (self.video_width, self.video_height))

            # Initialize IMU file
            if self.config.save_imu_data and (self.accel_stream_profile or self.gyro_stream_profile):
                imu_path = os.path.join(self.config.output_imu_dir, "imu_data.csv")
                imu_file = open(imu_path, 'w')
                imu_file.write("timestamp_ms,stream_type,x,y,z\n")

            # Main processing loop
            while True:
                try:
                    frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                except RuntimeError:
                    print("\nEnd of file reached.")
                    break

                aligned_frames = self.align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                
                if color_frame and depth_frame:
                    processed_color_frames_count += 1
                    pbar.update(1)
                    timestamp_ms = int(color_frame.get_timestamp())
                    color_image = cv2.cvtColor(np.asanyarray(color_frame.get_data()), cv2.COLOR_RGB2BGR)
                    depth_raw = np.asanyarray(depth_frame.get_data())
                    
                    if self.config.save_individual_frames:
                        cv2.imwrite(os.path.join(self.config.output_color_frames_dir, f"color_{timestamp_ms:013d}.png"), color_image)
                        np.save(os.path.join(self.config.output_depth_raw_frames_dir, f"depth_raw_{timestamp_ms:013d}.npy"), depth_raw)
                    
                    if color_video_writer: color_video_writer.write(color_image)
                    if depth_video_writer: 
                        depth_viz = cv2.applyColorMap(cv2.convertScaleAbs(depth_raw, alpha=0.03), cv2.COLORMAP_JET)
                        if self.config.save_individual_frames:
                            cv2.imwrite(os.path.join(self.config.output_depth_viz_frames_dir, f"depth_viz_{timestamp_ms:013d}.png"), depth_viz)
                        depth_video_writer.write(depth_viz)

                if imu_file:
                    for frame in frames:
                        if frame.is_motion_frame():
                            motion = frame.as_motion_frame()
                            stream_type = "Accel" if motion.get_profile().stream_type() == rs.stream.accel else "Gyro"
                            imu_file.write(f"{motion.get_timestamp():.6f},{stream_type},{motion.get_motion_data().x:.6f},{motion.get_motion_data().y:.6f},{motion.get_motion_data().z:.6f}\n")
        
        finally:
            if pbar: pbar.close()
            if color_video_writer: color_video_writer.release()
            if depth_video_writer: depth_video_writer.release()
            if imu_file: imu_file.close()
            self.pipeline.stop()
            print(f"\nProcessing finished. {processed_color_frames_count} frame pairs processed.")
            print(f"Outputs saved in: {os.path.abspath(self.config.output_base_dir)}")