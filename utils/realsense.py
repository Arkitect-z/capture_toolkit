import pyrealsense2 as rs
import numpy as np
import cv2
import os
import shutil
import datetime # For getting bag file duration
from tqdm import tqdm # For progress bar

class RealSenseBagProcessor:
    """
    A class for processing Intel RealSense .bag files.
    It can extract, align, and save RGB and depth image data (as individual files and videos),
    and extract and save IMU data.
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
                 # Optionally raise an exception or allow continuation with some features disabled


    def _create_output_dirs(self, *dirs):
        """Creates output directories. Does not clear them by default for video/IMU."""
        for d in dirs:
            # if os.path.exists(d): # Uncomment if you prefer to clear these too
                # print(f"Clearing existing directory: {d}")
                # shutil.rmtree(d)
                # os.makedirs(d, exist_ok=True)
            os.makedirs(d, exist_ok=True) # Creates if not exists, does nothing if it exists
            print(f"Output directory ensured: {d}")


    def process_and_save_all(self, 
                             output_dir_color_frames, 
                             output_dir_depth_raw_frames, 
                             output_dir_depth_viz_frames,
                             output_video_dir, # New: Video output directory
                             output_imu_dir,   # New: IMU data output directory
                             save_individual_frames=True,
                             save_video=True,
                             save_imu_data=True,
                             depth_colormap_enabled=True,
                             video_codec='mp4v'): # e.g., 'mp4v' (H.264 for .mp4), 'XVID' (for .avi)
        """
        Processes the .bag file, saving individual frames, videos, and IMU data.
        """
        self._initialize_streams_and_get_info() # Get stream info first

        # Create directories for individual frames (and clear them if they exist)
        if save_individual_frames:
            for d in [output_dir_color_frames, output_dir_depth_raw_frames, output_dir_depth_viz_frames]:
                if os.path.exists(d):
                    print(f"Clearing existing directory for individual frames: {d}")
                    shutil.rmtree(d)
                os.makedirs(d, exist_ok=True)
                print(f"Output directory for individual frames ensured: {d}")
        
        self._create_output_dirs(output_video_dir, output_imu_dir) # Ensure video/IMU dirs exist

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
                print("Warning: Could not get playback control object.")
                pbar = tqdm(desc="Processing frames", unit="frame") # Total unknown

            # Initialize VideoWriters (if video saving is enabled and stream info is available)
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


            # Initialize IMU file (if enabled)
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
                    # Wait for a coherent set of frames, with a timeout
                    frames = self.pipeline.wait_for_frames(timeout_ms=10000) # 10 second timeout
                except RuntimeError as e:
                    if "Frame didn't arrive within" in str(e).lower() or "end of file" in str(e).lower():
                        print("\nReached end of file or frame wait timeout.")
                        break # Exit loop
                    else:
                        # print(f"\nRuntime error while waiting for frames: {e}") # Can be too verbose
                        if playback and playback.current_status() == rs.playback_status.stopped:
                            print("\nPlayback stopped (end of file).")
                            break
                        continue # Try next frame

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
                    color_image = np.asanyarray(color_frame.get_data())
                    depth_image_raw = np.asanyarray(depth_frame.get_data()) # uint16 raw data
                    timestamp_ms = int(color_frame.get_timestamp()) # Use color frame's timestamp for aligned pair

                    if save_individual_frames:
                        color_filename = os.path.join(output_dir_color_frames, f"color_{timestamp_ms:013d}.png")
                        cv2.imwrite(color_filename, color_image)
                        
                        depth_raw_filename = os.path.join(output_dir_depth_raw_frames, f"depth_raw_{timestamp_ms:013d}.npy")
                        np.save(depth_raw_filename, depth_image_raw)
                        
                        # Apply colormap for visualized depth frame
                        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_raw, alpha=0.03), cv2.COLORMAP_JET)
                        depth_viz_filename = os.path.join(output_dir_depth_viz_frames, f"depth_viz_{timestamp_ms:013d}.png")
                        cv2.imwrite(depth_viz_filename, depth_colormap)
                    
                    if save_video and color_video_writer and depth_video_writer:
                        color_video_writer.write(color_image)
                        # Ensure depth_colormap is available for video even if individual frames aren't saved with it
                        if 'depth_colormap' not in locals() or not depth_colormap_enabled:
                             depth_colormap_for_video = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_raw, alpha=0.03), cv2.COLORMAP_JET)
                        else:
                             depth_colormap_for_video = depth_colormap # Use already computed one if available
                        depth_video_writer.write(depth_colormap_for_video)
                    
                    frame_count_for_pbar_update +=1 # Increment for progress bar only on successful image pair

                # --- Process IMU frames ---
                if save_imu_data and imu_file:
                    for frame in frames: # Iterate through all frames in the frameset
                        if frame.is_motion_frame():
                            motion = frame.as_motion_frame()
                            m_profile = motion.get_profile()
                            ts = motion.get_timestamp() # double, milliseconds
                            data = motion.get_motion_data() # rs.vector (x,y,z)
                            
                            stream_type_name = "UnknownMotion" # Default
                            if m_profile.stream_type() == rs.stream.accel:
                                stream_type_name = "Accel"
                            elif m_profile.stream_type() == rs.stream.gyro:
                                stream_type_name = "Gyro"
                            
                            imu_file.write(f"{ts:.6f},{stream_type_name},{data.x:.6f},{data.y:.6f},{data.z:.6f}\n")
                
                if pbar and frame_count_for_pbar_update > 0:
                    pbar.update(frame_count_for_pbar_update)
                    frame_count_for_pbar_update = 0
                elif pbar and not (color_frame and depth_frame): # If no image frames but loop continues (e.g. for IMU)
                    pbar.update(0) # Keep progress bar active


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
    BAG_FILE = "RealSense/20250528_000356.bag" # <--- Modify to your .bag file path (Linux/MacOS example)

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