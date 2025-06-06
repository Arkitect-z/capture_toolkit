import pyrealsense2 as rs
import numpy as np
import cv2
import os
import shutil
import datetime
from tqdm import tqdm

class RealSenseBagProcessor:
    """
    A class for processing Intel RealSense .bag files.
    It can extract, align, and save RGB and depth image data (as individual files and videos),
    extract and save IMU data, and save stream specification details.
    """
    def __init__(self, bag_file_path):
        """
        Initializes the RealSenseBagProcessor.

        Args:
            bag_file_path (str): Path to the .bag file.
        """
        if not os.path.exists(bag_file_path):
            raise FileNotFoundError(f".bag file not found: {bag_file_path}")
        
        self.bag_file_path = bag_file_path
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        rs.config.enable_device_from_file(self.config, self.bag_file_path, repeat_playback=False)
        
        self.align_to_stream = rs.stream.color
        self.align = rs.align(self.align_to_stream)

        self.profile = None
        self.color_stream_profile = None
        self.depth_stream_profile = None
        self.accel_stream_profile = None
        self.gyro_stream_profile = None

        self.video_width = 0
        self.video_height = 0
        self.video_fps = 30.0 # Default value, will be overridden

    def _initialize_streams_and_get_info(self):
        """
        Initializes the pipeline configuration phase (without starting playback) 
        to obtain stream information for setup. This helps get video parameters 
        and estimate total frames.
        """
        try:
            self.profile = self.config.resolve(self.pipeline) # Resolve config but don't start
            if self.profile:
                # Get video stream info
                color_profile = self.profile.get_stream(rs.stream.color)
                if color_profile:
                    self.color_stream_profile = color_profile.as_video_stream_profile()
                    self.video_width = self.color_stream_profile.width()
                    self.video_height = self.color_stream_profile.height()
                    self.video_fps = float(self.color_stream_profile.fps())
                    print(f"Video stream info: {self.video_width}x{self.video_height} @ {self.video_fps:.2f} FPS")
                else:
                    print("Warning: Color stream not found in the .bag file. Video export will be affected.")

                depth_profile = self.profile.get_stream(rs.stream.depth)
                if depth_profile:
                    self.depth_stream_profile = depth_profile.as_video_stream_profile()
                else:
                    print("Warning: Depth stream not found in the .bag file.")

                # Check IMU streams
                accel_profile = self.profile.get_stream(rs.stream.accel)
                if accel_profile:
                    self.accel_stream_profile = accel_profile.as_motion_stream_profile()
                    print(f"Found Accelerometer stream (FPS: {self.accel_stream_profile.fps()})")
                else:
                    print("Accelerometer stream not found in the .bag file.")

                gyro_profile = self.profile.get_stream(rs.stream.gyro)
                if gyro_profile:
                    self.gyro_stream_profile = gyro_profile.as_motion_stream_profile()
                    print(f"Found Gyroscope stream (FPS: {self.gyro_stream_profile.fps()})")
                else:
                    print("Gyroscope stream not found in the .bag file.")
            else:
                raise RuntimeError("Could not resolve pipeline configuration from .bag file.")

        except Exception as e:
            print(f"Error during stream info initialization: {e}")
            if not self.color_stream_profile:
                print("Error: Could not get color stream info. Video export and frame count estimation might fail.")

    def _create_output_dirs(self, *dirs):
        """Creates output directories."""
        for d in dirs:
            os.makedirs(d, exist_ok=True)
            print(f"Output directory ensured: {d}")

    def _write_intrinsics(self, file, profile):
        """Helper function to write stream intrinsics to a file."""
        intrinsics = profile.get_intrinsics()
        file.write(f"  Intrinsics:\n")
        file.write(f"    - fx (focal length x): {intrinsics.fx}\n")
        file.write(f"    - fy (focal length y): {intrinsics.fy}\n")
        file.write(f"    - ppx (principal point x): {intrinsics.ppx}\n")
        file.write(f"    - ppy (principal point y): {intrinsics.ppy}\n")
        file.write(f"    - width: {intrinsics.width}\n")
        file.write(f"    - height: {intrinsics.height}\n")
        file.write(f"    - Distortion Model: {intrinsics.model}\n")
        file.write(f"    - Distortion Coeffs: {intrinsics.coeffs}\n")

    def _save_stream_specifications(self, output_spec_dir):
        """Saves RGB and Depth stream specifications to a text file."""
        spec_file_path = os.path.join(output_spec_dir, "stream_specifications.txt")
        print(f"Saving stream specifications to: {spec_file_path}")
        with open(spec_file_path, 'w') as f:
            f.write(f"Specifications for: {os.path.basename(self.bag_file_path)}\n")
            f.write(f"Processed on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*40 + "\n\n")

            if self.color_stream_profile:
                f.write("--- RGB Color Stream Specs ---\n")
                f.write(f"- Resolution: {self.color_stream_profile.width()} x {self.color_stream_profile.height()}\n")
                f.write(f"- FPS: {self.color_stream_profile.fps()}\n")
                f.write(f"- Format: {self.color_stream_profile.format()}\n")
                self._write_intrinsics(f, self.color_stream_profile)
                f.write("\n")

            if self.depth_stream_profile:
                f.write("--- Depth Stream Specs ---\n")
                depth_sensor = self.profile.get_device().first_depth_sensor()
                depth_scale = depth_sensor.get_depth_scale()
                f.write(f"- Resolution: {self.depth_stream_profile.width()} x {self.depth_stream_profile.height()}\n")
                f.write(f"- FPS: {self.depth_stream_profile.fps()}\n")
                f.write(f"- Format: {self.depth_stream_profile.format()}\n")
                f.write(f"- Depth Scale: {depth_scale:.6f} (Multiply raw depth value by this to get meters)\n")
                self._write_intrinsics(f, self.depth_stream_profile)
                f.write("\n")
        print("Stream specifications saved.")

    def process_and_save_all(self, 
                             output_dir_color_frames, 
                             output_dir_depth_raw_frames, 
                             output_dir_depth_viz_frames,
                             output_video_dir,
                             output_imu_dir,
                             output_spec_dir, # New: Spec file output directory
                             save_individual_frames=True,
                             save_video=True,
                             save_imu_data=True,
                             depth_colormap_enabled=True,
                             video_codec='mp4v'):
        """
        Processes the .bag file, saving individual frames, videos, IMU data, and specs.
        """
        self._initialize_streams_and_get_info() # Get stream info first

        # Create directories for individual frames (and clear them if they exist)
        if save_individual_frames:
            for d in [output_dir_color_frames, output_dir_depth_raw_frames, output_dir_depth_viz_frames]:
                if os.path.exists(d):
                    shutil.rmtree(d)
                os.makedirs(d)
                print(f"Output directory for individual frames ensured: {d}")
        
        self._create_output_dirs(output_video_dir, output_imu_dir, output_spec_dir) # Ensure video/IMU/spec dirs exist

        # Save stream specification file
        self._save_stream_specifications(output_spec_dir)

        color_video_writer = None
        depth_video_writer = None
        imu_file = None
        pbar = None
        processed_color_frames_count = 0

        try:
            print(f"Starting processing of .bag file: {self.bag_file_path}")
            self.profile = self.pipeline.start(self.config) # Now start the pipeline for playback
            
            playback = self.profile.get_device().as_playback()
            if playback:
                playback.set_real_time(False) # Process as fast as possible
                duration_timedelta = playback.get_duration()
                if duration_timedelta.total_seconds() > 0 and self.video_fps > 0:
                    total_frames_estimate = int(duration_timedelta.total_seconds() * self.video_fps)
                    pbar = tqdm(total=total_frames_estimate, desc="Processing frames", unit="frame")
                else:
                    pbar = tqdm(desc="Processing frames", unit="frame") # Total unknown
            else:
                pbar = tqdm(desc="Processing frames", unit="frame") # Total unknown

            # Initialize VideoWriters
            if save_video and self.color_stream_profile and self.video_width > 0 and self.video_height > 0:
                video_extension = 'mp4' if video_codec.lower() in ['mp4v', 'h264', 'avc1'] else 'avi'
                color_video_path = os.path.join(output_video_dir, f"rgb_video.{video_extension}")
                depth_video_path = os.path.join(output_video_dir, f"depth_visualized_video.{video_extension}")
                
                fourcc = cv2.VideoWriter_fourcc(*video_codec)
                color_video_writer = cv2.VideoWriter(color_video_path, fourcc, self.video_fps, (self.video_width, self.video_height))
                depth_video_writer = cv2.VideoWriter(depth_video_path, fourcc, self.video_fps, (self.video_width, self.video_height))
                print(f"Will save RGB video to: {color_video_path}")
                print(f"Will save visualized depth video to: {depth_video_path}")
            elif save_video:
                print("Warning: Video saving disabled due to missing color stream info or invalid video dimensions.")
                save_video = False

            # Initialize IMU file
            if save_imu_data and (self.accel_stream_profile or self.gyro_stream_profile):
                imu_file_path = os.path.join(output_imu_dir, "imu_data.csv")
                imu_file = open(imu_file_path, 'w')
                imu_file.write("timestamp_ms,stream_type,x,y,z\n") # CSV header
                print(f"Will save IMU data to: {imu_file_path}")
            elif save_imu_data:
                print("Warning: IMU streams not found, IMU data saving disabled.")
                save_imu_data = False

            frame_count_for_pbar_update = 0

            while True:
                try:
                    frames = self.pipeline.wait_for_frames(timeout_ms=10000)
                except RuntimeError as e:
                    if "end of file" in str(e).lower() or "frame didn't arrive" in str(e).lower():
                        print("\nReached end of file or frame wait timeout.")
                        break
                    else:
                        if playback and playback.current_status() == rs.playback_status.stopped:
                            print("\nPlayback stopped (end of file).")
                            break
                        continue

                if not frames:
                    if playback and playback.current_status() == rs.playback_status.stopped:
                        print("\nPlayback stopped (end of file).")
                        break
                    continue
                
                # --- Process image frames ---
                aligned_frames = self.align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                
                if color_frame and depth_frame:
                    processed_color_frames_count += 1
                    color_image_rgb = np.asanyarray(color_frame.get_data())
                    
                    # === COLOR FIX: Convert RGB (from RealSense) to BGR (for OpenCV) ===
                    color_image_bgr = cv2.cvtColor(color_image_rgb, cv2.COLOR_RGB2BGR)
                    # =====================================================================

                    depth_image_raw = np.asanyarray(depth_frame.get_data()) # uint16 raw data
                    timestamp_ms = int(color_frame.get_timestamp())

                    # Apply colormap for visualized depth frame
                    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_raw, alpha=0.03), cv2.COLORMAP_JET)

                    if save_individual_frames:
                        color_filename = os.path.join(output_dir_color_frames, f"color_{timestamp_ms:013d}.png")
                        cv2.imwrite(color_filename, color_image_bgr) # <-- Use BGR image
                        
                        depth_raw_filename = os.path.join(output_dir_depth_raw_frames, f"depth_raw_{timestamp_ms:013d}.npy")
                        np.save(depth_raw_filename, depth_image_raw)
                        
                        depth_viz_filename = os.path.join(output_dir_depth_viz_frames, f"depth_viz_{timestamp_ms:013d}.png")
                        cv2.imwrite(depth_viz_filename, depth_colormap)
                    
                    if save_video and color_video_writer and depth_video_writer:
                        color_video_writer.write(color_image_bgr) # <-- Use BGR image
                        depth_video_writer.write(depth_colormap)
                    
                    frame_count_for_pbar_update += 1

                # --- Process IMU frames ---
                if save_imu_data and imu_file:
                    for frame in frames: # Iterate through all frames in the frameset
                        if frame.is_motion_frame():
                            motion = frame.as_motion_frame()
                            m_profile = motion.get_profile()
                            ts = motion.get_timestamp()
                            data = motion.get_motion_data()
                            
                            stream_type_name = "UnknownMotion" # Default
                            if m_profile.stream_type() == rs.stream.accel:
                                stream_type_name = "Accel"
                            elif m_profile.stream_type() == rs.stream.gyro:
                                stream_type_name = "Gyro"
                            
                            imu_file.write(f"{ts:.6f},{stream_type_name},{data.x:.6f},{data.y:.6f},{data.z:.6f}\n")
                
                if pbar and frame_count_for_pbar_update > 0:
                    pbar.update(frame_count_for_pbar_update)
                    frame_count_for_pbar_update = 0

        except Exception as e:
            print(f"\nAn error occurred during processing: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if pbar:
                pbar.close()
            if color_video_writer:
                color_video_writer.release()
                print(f"\nRGB video saved.")
            if depth_video_writer:
                depth_video_writer.release()
                print(f"Visualized depth video saved.")
            if imu_file:
                imu_file.close()
                print(f"IMU data saved.")
            
            self.pipeline.stop()
            print(f"Processing finished. A total of {processed_color_frames_count} valid color/depth frame pairs were processed.")
            if save_individual_frames:
                print(f"Individual color frames saved to: {os.path.abspath(output_dir_color_frames)}")
                print(f"Individual raw depth data saved to: {os.path.abspath(output_dir_depth_raw_frames)}")
                print(f"Individual visualized depth frames saved to: {os.path.abspath(output_dir_depth_viz_frames)}")

# --- Test Code ---
if __name__ == "__main__":
    # ================================================================
    # Configuration Parameters - Please modify these paths and options
    # ================================================================
    # BAG_FILE = r"C:\path\to\your\realsense_recording.bag" # Windows example
    BAG_FILE = "data_pilot/RealSense/20250528_000356.bag" # <--- Modify to your .bag file path (Linux/MacOS example)

    # Base output directory
    OUTPUT_BASE_DIR = "realsense_export"
    
    # Output directories for individual frames (if save_individual_frames = True)
    OUTPUT_COLOR_FRAMES_DIR = os.path.join(OUTPUT_BASE_DIR, "frames", "color")
    OUTPUT_DEPTH_RAW_FRAMES_DIR = os.path.join(OUTPUT_BASE_DIR, "frames", "depth_raw_npy")
    OUTPUT_DEPTH_VIZ_FRAMES_DIR = os.path.join(OUTPUT_BASE_DIR, "frames", "depth_visualized_png")

    # Output directory for video files (if save_video = True)
    OUTPUT_VIDEO_DIR = os.path.join(OUTPUT_BASE_DIR, "videos")

    # Output directory for IMU data (if save_imu_data = True)
    OUTPUT_IMU_DIR = os.path.join(OUTPUT_BASE_DIR, "imu_data")

    # New: Output directory for the stream specifications file
    OUTPUT_SPEC_DIR = os.path.join(OUTPUT_BASE_DIR, "specifications")
    # ================================================================

    # Check if the .bag file exists
    if not os.path.exists(BAG_FILE):
        print(f"Error: Please change the 'BAG_FILE' variable to the actual path of your .bag file.")
        print(f"Current BAG_FILE is set to: {BAG_FILE}")
        exit()
    else:
        print(f"Using .bag file: {os.path.abspath(BAG_FILE)}")

    try:
        processor = RealSenseBagProcessor(bag_file_path=BAG_FILE)
        
        processor.process_and_save_all(
            output_dir_color_frames=OUTPUT_COLOR_FRAMES_DIR,
            output_dir_depth_raw_frames=OUTPUT_DEPTH_RAW_FRAMES_DIR,
            output_dir_depth_viz_frames=OUTPUT_DEPTH_VIZ_FRAMES_DIR,
            output_video_dir=OUTPUT_VIDEO_DIR,
            output_imu_dir=OUTPUT_IMU_DIR,
            output_spec_dir=OUTPUT_SPEC_DIR, # Pass the new directory
            save_individual_frames=True, # Set to True or False
            save_video=True,             # Set to True or False
            save_imu_data=True,          # Set to True or False
            depth_colormap_enabled=True,
            video_codec='mp4v'           # 'mp4v' for .mp4, 'XVID' for .avi, 'avc1' for H.264 .mp4
        )
        
        print("\nScript execution finished.")

    except FileNotFoundError as fnf_error:
        print(fnf_error)
    except Exception as e:
        print(f"An unexpected error occurred during test code execution: {e}")
        import traceback
        traceback.print_exc()