# tactile_processor.py
import pandas as pd

# 该映射基于 "采集设备手指对应.jpg" 图片。
# 它将25个传感器ID映射到21个MANO关键点。
# 手掌上的传感器21-25将被平均处理以代表“手腕”关键点。
MANO_MAPPING = {
    # 手指
    'Thumb_Tip': 'Sensor_1', 'Thumb_IP': 'Sensor_2', 'Thumb_MCP': 'Sensor_3', 'Thumb_CMC': 'Sensor_4',
    'Index_Tip': 'Sensor_5', 'Index_DIP': 'Sensor_6', 'Index_PIP': 'Sensor_7', 'Index_MCP': 'Sensor_8',
    'Middle_Tip': 'Sensor_9', 'Middle_DIP': 'Sensor_10', 'Middle_PIP': 'Sensor_11', 'Middle_MCP': 'Sensor_12',
    'Ring_Tip': 'Sensor_13', 'Ring_DIP': 'Sensor_14', 'Ring_PIP': 'Sensor_15', 'Ring_MCP': 'Sensor_16',
    'Pinky_Tip': 'Sensor_17', 'Pinky_DIP': 'Sensor_18', 'Pinky_PIP': 'Sensor_19', 'Pinky_MCP': 'Sensor_20',
    # 手掌传感器，用于平均计算手腕数据
    'Wrist': ['Sensor_21', 'Sensor_22', 'Sensor_23', 'Sensor_24', 'Sensor_25']
}

def load_and_process_tactile_data(csv_file_path):
    """
    从CSV文件加载触觉数据，进行处理，并将其映射到MANO手部模型。
    该函数可以处理因行尾逗号而导致的多余空列（26列）的CSV文件。

    Args:
        csv_file_path (str or file-like object): CSV文件的路径或文件对象。

    Returns:
        pandas.DataFrame: 一个包含时间戳和MANO关键点压力值的DataFrame。
                         如果文件无效，则返回None。
    """
    try:
        df_raw = pd.read_csv(csv_file_path, header=None)
        
        num_columns = len(df_raw.columns)
        
        if num_columns == 26:
            df_sensors = df_raw.iloc[:, 0:25]
        elif num_columns == 25:
            df_sensors = df_raw
        else:
            raise ValueError(f"预期的列数为25或26，但得到了 {num_columns}")

        if len(df_sensors.columns) != 25:
             raise ValueError(f"数据处理后得到 {len(df_sensors.columns)} 列, 预期为25列。")

        df_sensors.columns = [f'Sensor_{i+1}' for i in range(25)]

        # --- FIX STARTS HERE ---
        # MODIFIED: 强制将所有传感器列转换为数值类型
        # 这可以防止因CSV中存在非数字字符而引发的TypeError。
        for col in df_sensors.columns:
            df_sensors[col] = pd.to_numeric(df_sensors[col], errors='coerce')

        # 将转换失败产生的NaN值替换为0
        df_sensors.fillna(0, inplace=True)
        # --- FIX ENDS HERE ---

        # 添加时间戳列
        df_sensors['timestamp'] = [i * 0.2 for i in range(len(df_sensors))]

    except Exception as e:
        print(f"读取或解析CSV时出错: {e}")
        return None

    # 创建一个新的DataFrame来存储MANO格式的数据
    df_mano = pd.DataFrame()
    df_mano['timestamp'] = df_sensors['timestamp']

    # 映射手指传感器
    for mano_joint, sensor_col in MANO_MAPPING.items():
        if mano_joint != 'Wrist':
            df_mano[mano_joint] = df_sensors[sensor_col]

    # 平均手掌传感器数据作为手腕数据
    wrist_sensors = MANO_MAPPING['Wrist']
    df_mano['Wrist'] = df_sensors[wrist_sensors].mean(axis=1)

    return df_mano

