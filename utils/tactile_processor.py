import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# This mapping maps 25 sensor IDs to 21 MANO keypoints.
# Sensors 21-25 on the palm will be averaged to represent the "Wrist" keypoint.
MANO_MAPPING = {
    # Fingers
    'Thumb_Tip': 'Sensor_1', 'Thumb_IP': 'Sensor_2', 'Thumb_MCP': 'Sensor_3', 'Thumb_CMC': 'Sensor_4',
    'Index_Tip': 'Sensor_5', 'Index_DIP': 'Sensor_6', 'Index_PIP': 'Sensor_7', 'Index_MCP': 'Sensor_8',
    'Middle_Tip': 'Sensor_9', 'Middle_DIP': 'Sensor_10', 'Middle_PIP': 'Sensor_11', 'Middle_MCP': 'Sensor_12',
    'Ring_Tip': 'Sensor_13', 'Ring_DIP': 'Sensor_14', 'Ring_PIP': 'Sensor_15', 'Ring_MCP': 'Sensor_16',
    'Pinky_Tip': 'Sensor_17', 'Pinky_DIP': 'Sensor_18', 'Pinky_PIP': 'Sensor_19', 'Pinky_MCP': 'Sensor_20',
    # Palm sensors, averaged for the wrist
    'Wrist': ['Sensor_21', 'Sensor_22', 'Sensor_23', 'Sensor_24', 'Sensor_25']
}

class TactileConfiguration:
    """
    A class to hold all configuration parameters for tactile data processing.
    """
    def __init__(self,
                 input_path: str,
                 output_folder: str,
                 output_filename: str = "tactile_120hz.csv",
                 original_fps: int = 5,
                 target_fps: int = 120):
        """
        Initializes the configuration.
        
        Args:
            input_path (str): The path to the input CSV file.
            output_folder (str): The directory to save the output file.
            output_filename (str): The name of the output file.
            original_fps (int): The original sampling rate of the data (Hz).
            target_fps (int): The target upsampling rate (Hz).
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: '{input_path}'")
            
        self.input_path = input_path
        self.output_folder = output_folder
        self.output_filename = output_filename
        self.original_fps = original_fps
        self.target_fps = target_fps
        
        # Derived output path
        self.output_filepath = os.path.join(self.output_folder, self.output_filename)


class TactileProcessor:
    """
    A processor class for handling tactile data.
    It encapsulates data loading, upsampling, and saving into distinct methods.
    """
    def __init__(self, config: TactileConfiguration):
        """
        Initializes the processor.
        """
        self.config = config
        self._setup_environment()

    def _setup_environment(self):
        """Creates the output directory."""
        if not os.path.exists(self.config.output_folder):
            os.makedirs(self.config.output_folder)

    def _load_and_process_raw_data(self):
        """
        Loads raw tactile data from a CSV file, processes it, and maps it to the MANO hand model.
        
        Returns:
            pd.DataFrame: A DataFrame containing timestamps and MANO keypoint pressure values.
        """
        try:
            # Load raw data, handling potential trailing commas
            df_raw = pd.read_csv(self.config.input_path, header=None)
            num_columns = len(df_raw.columns)
            
            if num_columns >= 25:
                # Take only the first 25 sensor columns
                df_sensors = df_raw.iloc[:, 0:25]
            else:
                raise ValueError(f"Expected at least 25 columns, but got {num_columns}")

            df_sensors.columns = [f'Sensor_{i+1}' for i in range(25)]

            # Force all sensor columns to numeric type, coercing errors to NaN
            for col in df_sensors.columns:
                df_sensors[col] = pd.to_numeric(df_sensors[col], errors='coerce')
            
            # Fill any resulting NaN values from conversion with 0
            df_sensors.fillna(0, inplace=True)

            # Add a timestamp column based on the original FPS
            df_sensors['timestamp'] = np.arange(len(df_sensors)) / self.config.original_fps

            # Create a new DataFrame to store the MANO-formatted data
            df_mano = pd.DataFrame()
            df_mano['timestamp'] = df_sensors['timestamp']

            # Map finger sensors
            for mano_joint, sensor_col in MANO_MAPPING.items():
                if mano_joint != 'Wrist':
                    df_mano[mano_joint] = df_sensors[sensor_col]

            # Average palm sensor data for the wrist data
            wrist_sensors = MANO_MAPPING['Wrist']
            df_mano['Wrist'] = df_sensors[wrist_sensors].mean(axis=1)

            return df_mano

        except Exception as e:
            print(f"Error reading or parsing CSV '{self.config.input_path}': {e}")
            raise

    def _upsample_data(self, df: pd.DataFrame):
        """
        Upsamples the time series data in the DataFrame to a target frequency.

        Args:
            df (pd.DataFrame): The input DataFrame, containing a 'timestamp' column.

        Returns:
            pd.DataFrame: The upsampled DataFrame.
        """
        if df.empty:
            return df
            
        # Set 'timestamp' as the index to perform time series resampling
        df_timed = df.set_index(pd.to_timedelta(df['timestamp'], unit='s'))
        
        # Create a new, higher-frequency time index
        time_step = 1 / self.config.target_fps
        new_time_index = np.arange(df_timed.index.total_seconds().min(), df_timed.index.total_seconds().max(), time_step)
        new_time_index_td = pd.to_timedelta(new_time_index, unit='s')

        # Resample and use linear interpolation to fill data for new time points
        df_resampled = df_timed.reindex(df_timed.index.union(new_time_index_td)).interpolate(method='linear').reindex(new_time_index_td)
        
        # Reset the index, moving the TimedeltaIndex into a column named 'index'
        df_resampled = df_resampled.reset_index()
        
        # FIX: Directly convert the 'index' column (Timedelta type) to total seconds and assign to 'timestamp'.
        # This overwrites or creates the 'timestamp' column, avoiding potential naming conflicts.
        df_resampled['timestamp'] = df_resampled['index'].dt.total_seconds()
        
        # Drop the temporary 'index' column
        df_resampled = df_resampled.drop(columns=['index'])

        return df_resampled

    def process_and_save(self):
        """
        Executes the full data processing pipeline: load, upsample, and save.
        """
        print(f"Processing: {self.config.input_path}")
        # 1. Load and process the raw data
        df_mano = self._load_and_process_raw_data()
        
        # 2. Upsample to the target frequency
        print(f"Upsampling data from {self.config.original_fps}Hz to {self.config.target_fps}Hz...")
        df_upsampled = self._upsample_data(df_mano)
        
        # 3. Save the processed file
        df_upsampled.to_csv(self.config.output_filepath, index=False, float_format='%.4f')
        print(f"Data successfully saved to: {self.config.output_filepath}")

# This is the old function, its functionality is now replaced by the TactileProcessor class.
# It is kept temporarily for backward compatibility with run_tactile_webapp.py.
def load_and_process_tactile_data(csv_file_path):
    """
    Loads tactile data from a CSV file, processes it, and maps it to the MANO hand model.
    This function can handle CSV files with an extra empty column (26 columns) due to trailing commas.

    Args:
        csv_file_path (str or file-like object): The path or a file-like object for the CSV file.

    Returns:
        pandas.DataFrame: A DataFrame with timestamps and MANO keypoint pressure values.
                         Returns None if the file is invalid.
    """
    try:
        df_raw = pd.read_csv(csv_file_path, header=None)
        
        num_columns = len(df_raw.columns)
        
        if num_columns >= 25:
            df_sensors = df_raw.iloc[:, 0:25]
        else:
            raise ValueError(f"Expected at least 25 columns, but got {num_columns}")

        df_sensors.columns = [f'Sensor_{i+1}' for i in range(25)]

        for col in df_sensors.columns:
            df_sensors[col] = pd.to_numeric(df_sensors[col], errors='coerce')

        df_sensors.fillna(0, inplace=True)
        
        # Default sampling rate is 5Hz
        df_sensors['timestamp'] = [i * 0.2 for i in range(len(df_sensors))]

    except Exception as e:
        print(f"Error reading or parsing CSV: {e}")
        return None

    df_mano = pd.DataFrame()
    df_mano['timestamp'] = df_sensors['timestamp']

    for mano_joint, sensor_col in MANO_MAPPING.items():
        if mano_joint != 'Wrist':
            df_mano[mano_joint] = df_sensors[sensor_col]

    wrist_sensors = MANO_MAPPING['Wrist']
    df_mano['Wrist'] = df_sensors[wrist_sensors].mean(axis=1)

    return df_mano
