# run_tactile_webapp.py
import base64
import io
import pandas as pd
import numpy as np
from zipstream import ZipStream
import dash
from dash import dcc, html, dash_table, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from flask import Response
import serial
import serial.tools.list_ports
import threading
import time
from collections import deque
from datetime import datetime
import csv
from tactile_processor import load_and_process_tactile_data, MANO_MAPPING

# --- 实时数据采集相关全局变量 ---
# 传感器尺寸
SENSOR_ROWS = 16
SENSOR_COLS = 16
NUM_POINTS = SENSOR_ROWS * SENSOR_COLS
PACKET_START_CHAR = 0x24 # '$'
PACKET_SIZE = 10 # 1 byte start, 1 byte ID, 8 bytes payload

# 实时数据存储
realtime_data_queue = deque(maxlen=200) # 存储最近200个数据点用于绘图
current_realtime_frame = np.zeros(25) # 存储映射到MANO的25个点
last_frame_time = 0
recorded_data = [] # 用于存储要保存的数据

# 串口相关
ser = None
serial_thread = None
is_reading = False
try:
    available_ports = [port.device for port in serial.tools.list_ports.comports()]
except Exception:
    available_ports = []

# 数据记录相关
is_recording = False

# --- 手部可视化设置 ---
MANO_2D_COORDS_LEFT = {
    'Wrist': {'x': 0.5, 'y': 0.1}, 'Thumb_CMC': {'x': 0.6, 'y': 0.25}, 'Thumb_MCP': {'x': 0.7, 'y': 0.35}, 'Thumb_IP': {'x': 0.8, 'y': 0.45}, 'Thumb_Tip': {'x': 0.9, 'y': 0.55},
    'Index_MCP': {'x': 0.55, 'y': 0.4}, 'Index_PIP': {'x': 0.55, 'y': 0.55}, 'Index_DIP': {'x': 0.55, 'y': 0.7}, 'Index_Tip': {'x': 0.55, 'y': 0.85},
    'Middle_MCP': {'x': 0.45, 'y': 0.42}, 'Middle_PIP': {'x': 0.45, 'y': 0.58}, 'Middle_DIP': {'x': 0.45, 'y': 0.74}, 'Middle_Tip': {'x': 0.45, 'y': 0.9},
    'Ring_MCP': {'x': 0.35, 'y': 0.4}, 'Ring_PIP': {'x': 0.35, 'y': 0.55}, 'Ring_DIP': {'x': 0.35, 'y': 0.7}, 'Ring_Tip': {'x': 0.35, 'y': 0.85},
    'Pinky_MCP': {'x': 0.25, 'y': 0.35}, 'Pinky_PIP': {'x': 0.27, 'y': 0.48}, 'Pinky_DIP': {'x': 0.29, 'y': 0.61}, 'Pinky_Tip': {'x': 0.31, 'y': 0.74}
}
MANO_2D_COORDS_RIGHT = {k: {'x': 1 - v['x'], 'y': v['y']} for k, v in MANO_2D_COORDS_LEFT.items()}
MANO_SKELETON_BONES = [
    ('Wrist', 'Thumb_CMC'), ('Thumb_CMC', 'Thumb_MCP'), ('Thumb_MCP', 'Thumb_IP'), ('Thumb_IP', 'Thumb_Tip'),
    ('Wrist', 'Index_MCP'), ('Index_MCP', 'Index_PIP'), ('Index_PIP', 'Index_DIP'), ('Index_DIP', 'Index_Tip'),
    ('Wrist', 'Middle_MCP'), ('Middle_MCP', 'Middle_PIP'), ('Middle_PIP', 'Middle_DIP'), ('Middle_DIP', 'Middle_Tip'),
    ('Wrist', 'Ring_MCP'), ('Ring_MCP', 'Ring_PIP'), ('Ring_PIP', 'Ring_DIP'), ('Ring_DIP', 'Ring_Tip'),
    ('Wrist', 'Pinky_MCP'), ('Pinky_MCP', 'Pinky_PIP'), ('Pinky_PIP', 'Pinky_DIP'), ('Pinky_DIP', 'Pinky_Tip'),
    ('Index_MCP', 'Middle_MCP'), ('Middle_MCP', 'Ring_MCP'), ('Ring_MCP', 'Pinky_MCP')
]

# --- 实时数据处理函数 ---
def map_raw_to_mano(raw_frame_flat):
    """将256个原始传感器数据点映射到25个MANO关键点"""
    # 此处为简化实现，直接使用MANO_MAPPING中的25个传感器ID
    # 假设 raw_frame_flat 已经是对应 MANO_MAPPING 的25个传感器的值
    df_sensors = pd.DataFrame([raw_frame_flat], columns=[f'Sensor_{i+1}' for i in range(25)])
    
    mano_values = {}
    for mano_joint, sensor_info in MANO_MAPPING.items():
        if isinstance(sensor_info, list): # 'Wrist'
            wrist_sensors = [col for col in sensor_info if col in df_sensors.columns]
            mano_values[mano_joint] = df_sensors[wrist_sensors].mean(axis=1).iloc[0]
        else: # Fingers
             if sensor_info in df_sensors.columns:
                mano_values[mano_joint] = df_sensors[sensor_info].iloc[0]
    
    # 按照MANO关键点顺序排列
    ordered_keys = list(MANO_2D_COORDS_LEFT.keys())
    return np.array([mano_values.get(key, 0) for key in ordered_keys])


def read_from_port():
    """在单独的线程中从串口读取和解析数据"""
    global ser, is_reading, current_realtime_frame, realtime_data_queue, last_frame_time, is_recording, recorded_data
    
    buffer = bytearray()
    data_accumulator = np.zeros(NUM_POINTS, dtype=int)

    while is_reading and ser and ser.is_open:
        try:
            if ser.in_waiting > 0:
                buffer.extend(ser.read(ser.in_waiting))
            
            while len(buffer) >= PACKET_SIZE:
                if buffer[0] == PACKET_START_CHAR:
                    packet_id = buffer[1]
                    
                    if packet_id <= 24 and packet_id % 8 == 0:
                        for i in range(8):
                            decoded_value = buffer[2 + i] - 40
                            data_accumulator[packet_id + i] = decoded_value

                        if packet_id == 24:
                            last_frame_time = time.time()
                            
                            # 注意：这里的 data_accumulator 是 256 个点
                            # 我们需要一个映射函数将其转换为25个MANO点
                            # 为了演示，我们先只取前25个点
                            mano_flat = data_accumulator[:25]
                            
                            current_realtime_frame = mano_flat # 更新全局变量
                            
                            # 更新绘图队列
                            max_val = np.max(current_realtime_frame)
                            realtime_data_queue.append({'time': last_frame_time, 'max_value': max_val})

                            if is_recording:
                                timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                                recorded_data.append([timestamp_str] + list(current_realtime_frame))
                        
                        del buffer[:PACKET_SIZE]
                    else:
                        del buffer[0]
                else:
                    del buffer[0]
            
            time.sleep(0.01)

        except Exception as e:
            print(f"读取或解析数据时出错: {e}")
            buffer.clear()
            time.sleep(0.01)

# --- App 初始化 ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
server = app.server
app.title = "双通道触觉数据可视化与实时采集"

# --- 辅助函数 ---
def make_hand_panel(hand_type, is_realtime=False):
    """创建左手或右手的UI面板"""
    title_text = "实时采集" if is_realtime else ("左手 (Left Hand)" if hand_type == 'left' else "右手 (Right Hand)")
    panel_id_prefix = f"realtime" if is_realtime else hand_type
    
    children = [
        dcc.Graph(id=f'hand-visualization-graph-{panel_id_prefix}', style={'height': '50vh'}),
        dash_table.DataTable(id=f'current-frame-table-{panel_id_prefix}', style_cell={'textAlign': 'left'}, style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'})
    ]
    
    if not is_realtime:
        children.insert(0, dcc.Upload(
            id=f'upload-data-{hand_type}',
            children=html.Div(['拖拽或 ', html.A('选择CSV文件')]),
            style={'width': '100%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'margin-bottom': '10px'},
        ))

    return dbc.Card([
        dbc.CardHeader(title_text),
        dbc.CardBody(children)
    ])

# --- App 布局 ---
app.layout = dbc.Container([
    # 非UI组件
    dcc.Store(id='processed-data-store-left'),
    dcc.Store(id='processed-data-store-right'),
    dcc.Store(id='play-state-store', data={'is_playing': False}),
    dcc.Store(id='pressure-range-store'),
    dcc.Interval(id='animation-interval', interval=200, n_intervals=0, disabled=True),
    dcc.Interval(id='realtime-interval', interval=100, n_intervals=0, disabled=True),
    dcc.Download(id="download-zip"),
    dcc.Download(id="download-realtime-zip"),

    # 页面标题
    dbc.Row(dbc.Col(html.H1("双通道触觉数据可视化与实时采集"), width=12), className="mt-4 mb-2 text-center"),
    
    # 主内容区
    dbc.Row([
        # 左侧面板: 文件可视化
        dbc.Col(make_hand_panel('left'), md=4),
        
        # 中间控制器
        dbc.Col(dbc.Card([
            dbc.CardHeader("同步与导出"),
            dbc.CardBody([
                html.H5("文件播放控制", className="font-weight-bold"),
                dcc.Slider(id='timeline-slider', min=0, max=0, value=0, step=1, marks=None, tooltip={"placement": "bottom", "always_visible": True}),
                html.Div([
                    dbc.Button("▶️ 播放", id="play-button", color="success", className="me-2"),
                    dbc.Button("⏸️ 暂停", id="pause-button", color="warning", className="me-2"),
                    dbc.Button("🔄 重置", id="reset-button", color="info"),
                ], className="mt-3 text-center"),
                html.Hr(),
                html.H5("可视化设置", className="font-weight-bold"),
                dbc.InputGroup([
                    dbc.InputGroupText("最大压力值"),
                    dbc.Input(id="manual-max-pressure-input", type="number", placeholder="自动"),
                ], className="mb-3"),
                dbc.Button("导出文件数据为ZIP", id="export-zip-button", color="primary", className="w-100"),
                html.Hr(),
                html.H5("实时采集控制", className="font-weight-bold"),
                dbc.InputGroup([
                    dbc.InputGroupText("串口"),
                    dcc.Dropdown(id='port-dropdown', options=[{'label': p, 'value': p} for p in available_ports], placeholder="选择串口", style={'flex': 1}),
                ]),
                dbc.Button('连接', id='connect-button', className="w-100 mt-2", color="success"),
                dbc.Button('断开', id='disconnect-button', className="w-100 mt-2", color="danger"),
                html.Div(id='status-message', className="text-center mt-2"),
                dbc.Button('开始记录', id='start-record-button', className="w-100 mt-2"),
                dbc.Button('停止记录', id='stop-record-button', className="w-100 mt-2", color="secondary"),
                html.Div(id='record-status', className="text-center mt-2"),
                 dbc.Button("导出实时数据为ZIP", id="export-realtime-button", color="info", className="w-100 mt-3"),
            ])
        ]), md=4),
        
        # 右侧面板: 实时数据可视化
        dbc.Col(make_hand_panel('right', is_realtime=True), md=4),
        
    ], className="mt-3"),
    
    # 增加一个独立的右手文件可视化面板
    dbc.Row([
        dbc.Col(md=4), # 占位
        dbc.Col(md=4), # 占位
        dbc.Col(make_hand_panel('right'), md=4, className="mt-3"),
    ],)
    
], fluid=True)


# --- 回调函数 (Callbacks) ---

# region 文件处理与播放控制回调
def parse_contents(contents):
    if contents is None: return None
    _, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    return load_and_process_tactile_data(io.StringIO(decoded.decode('utf-8')))

@app.callback(
    [Output('processed-data-store-left', 'data'),
     Output('processed-data-store-right', 'data'),
     Output('timeline-slider', 'max'),
     Output('pressure-range-store', 'data'),
     Output('manual-max-pressure-input', 'placeholder')],
    [Input('upload-data-left', 'contents'),
     Input('upload-data-right', 'contents')],
    [State('processed-data-store-left', 'data'),
     State('processed-data-store-right', 'data')],
    prevent_initial_call=True
)
def process_uploaded_files(contents_l, contents_r, existing_data_l, existing_data_r):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    df_l = pd.read_json(existing_data_l, orient='split') if existing_data_l else None
    df_r = pd.read_json(existing_data_r, orient='split') if existing_data_r else None

    if trigger_id == 'upload-data-left': df_l = parse_contents(contents_l)
    elif trigger_id == 'upload-data-right': df_r = parse_contents(contents_r)

    json_l = df_l.to_json(orient='split') if df_l is not None else None
    json_r = df_r.to_json(orient='split') if df_r is not None else None

    frames_l = len(df_l) if df_l is not None else 0
    frames_r = len(df_r) if df_r is not None else 0
    max_frames = max(frames_l, frames_r) if frames_l > 0 or frames_r > 0 else 0
    timeline_max = max(0, max_frames - 1)

    all_pressures = []
    if df_l is not None: all_pressures.append(df_l.drop('timestamp', axis=1))
    if df_r is not None: all_pressures.append(df_r.drop('timestamp', axis=1))

    if not all_pressures: return json_l, json_r, 0, None, "自动"

    combined_df = pd.concat(all_pressures)
    min_val = combined_df.min().min()
    p99_val = combined_df.quantile(0.995).quantile(0.995) if not combined_df.empty else 1.0
    range_data = {'min': min_val, 'p99': p99_val}
    placeholder_text = f"自动 (推荐: {p99_val:.2f})"
    
    return json_l, json_r, timeline_max, range_data, placeholder_text

@app.callback(
    [Output('animation-interval', 'disabled'),
     Output('play-state-store', 'data'),
     Output('timeline-slider', 'value', allow_duplicate=True)],
    [Input('play-button', 'n_clicks'),
     Input('pause-button', 'n_clicks'),
     Input('reset-button', 'n_clicks')],
    [State('play-state-store', 'data')],
    prevent_initial_call=True
)
def control_animation(play_clicks, pause_clicks, reset_clicks, state):
    button_id = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
    is_playing = state['is_playing']
    new_value = no_update
    if button_id == 'play-button': is_playing = True
    if button_id == 'pause-button': is_playing = False
    if button_id == 'reset-button':
        is_playing = False
        new_value = 0
    state['is_playing'] = is_playing
    return not is_playing, state, new_value

@app.callback(
    Output('timeline-slider', 'value'),
    Input('animation-interval', 'n_intervals'),
    [State('timeline-slider', 'value'), State('timeline-slider', 'max'), State('play-state-store', 'data')],
    prevent_initial_call=True
)
def advance_slider(n, current_value, max_value, play_state):
    if play_state['is_playing'] and current_value < max_value:
        return current_value + 1
    return no_update
# endregion

# region 可视化更新回调
def create_figure_from_file(df, frame_index, range_data, manual_max, hand_type):
    if df is None or frame_index >= len(df) or range_data is None:
        fig = go.Figure(layout={'xaxis': {'visible': False}, 'yaxis': {'visible': False}, 'title': f'无数据'})
        return fig, []

    coords = MANO_2D_COORDS_LEFT if hand_type == 'left' else MANO_2D_COORDS_RIGHT
    current_frame_data = df.iloc[frame_index]
    
    min_pressure = range_data.get('min', 0)
    max_pressure_for_color = manual_max if manual_max is not None and manual_max > min_pressure else range_data.get('p99', 1.0)
    pressure_range = max(1.0, max_pressure_for_color - min_pressure)

    joint_names = [col for col in df.columns if col != 'timestamp']
    points_x = [coords.get(j, {}).get('x') for j in joint_names]
    points_y = [coords.get(j, {}).get('y') for j in joint_names]
    pressures = [current_frame_data.get(j, 0) for j in joint_names]
    normalized_pressures = [np.clip((p - min_pressure) / pressure_range, 0, 1) for p in pressures]

    joint_points = go.Scatter(
        x=points_x, y=points_y, mode='markers',
        marker=dict(
            color=normalized_pressures, colorscale='RdYlBu_r', cmin=0, cmax=1,
            colorbar=dict(title=f"压力\n(0-{max_pressure_for_color:.0f})"),
            size=[10 + p * 40 for p in normalized_pressures], sizemin=4),
        text=[f"{name}<br>压力: {val:.2f}" for name, val in current_frame_data.items() if name != 'timestamp'], hoverinfo='text')

    lines_x, lines_y = [], []
    for bone_start, bone_end in MANO_SKELETON_BONES:
        if bone_start in coords and bone_end in coords:
            lines_x.extend([coords[bone_start]['x'], coords[bone_end]['x'], None])
            lines_y.extend([coords[bone_start]['y'], coords[bone_end]['y'], None])
    skeleton_lines = go.Scatter(x=lines_x, y=lines_y, mode='lines', line=dict(color='grey', width=2), hoverinfo='none')

    fig = go.Figure(data=[skeleton_lines, joint_points])
    fig.update_layout(
        title=f"帧: {frame_index}, 时间戳: {current_frame_data['timestamp']:.2f}s",
        xaxis=dict(visible=False, range=[0, 1]), yaxis=dict(visible=False, range=[0, 1.1], scaleanchor="x", scaleratio=1.2),
        showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
    
    table_data = current_frame_data.drop('timestamp').reset_index()
    table_data.columns = ['关节', '压力值']
    return fig, table_data.to_dict('records')

def create_figure_realtime(frame_data, range_data, manual_max):
    if frame_data is None or len(frame_data) == 0 or range_data is None:
        fig = go.Figure(layout={'xaxis': {'visible': False}, 'yaxis': {'visible': False}, 'title': '等待数据...'})
        return fig, []
        
    coords = MANO_2D_COORDS_RIGHT # 实时数据固定为右手
    
    min_pressure = 0
    max_pressure_for_color = manual_max if manual_max is not None and manual_max > 0 else 1024 # 实时数据默认范围
    pressure_range = max(1.0, max_pressure_for_color - min_pressure)
    
    joint_names = list(coords.keys())
    pressures = frame_data[:len(joint_names)] # 确保数据长度匹配
    normalized_pressures = [np.clip((p - min_pressure) / pressure_range, 0, 1) for p in pressures]

    points_x = [v['x'] for v in coords.values()]
    points_y = [v['y'] for v in coords.values()]
    
    joint_points = go.Scatter(
        x=points_x, y=points_y, mode='markers',
        marker=dict(
            color=normalized_pressures, colorscale='RdYlBu_r', cmin=0, cmax=1,
            colorbar=dict(title=f"压力\n(0-{max_pressure_for_color:.0f})"),
            size=[10 + p * 40 for p in normalized_pressures], sizemin=4),
        text=[f"{name}<br>压力: {val:.2f}" for name, val in zip(joint_names, pressures)], hoverinfo='text')

    lines_x, lines_y = [], []
    for bone_start, bone_end in MANO_SKELETON_BONES:
        if bone_start in coords and bone_end in coords:
            lines_x.extend([coords[bone_start]['x'], coords[bone_end]['x'], None])
            lines_y.extend([coords[bone_start]['y'], coords[bone_end]['y'], None])
    skeleton_lines = go.Scatter(x=lines_x, y=lines_y, mode='lines', line=dict(color='grey', width=2), hoverinfo='none')
    
    fig = go.Figure(data=[skeleton_lines, joint_points])
    fig.update_layout(
        title=f"实时数据",
        xaxis=dict(visible=False, range=[0, 1]), yaxis=dict(visible=False, range=[0, 1.1], scaleanchor="x", scaleratio=1.2),
        showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
        
    table_data = pd.DataFrame({'关节': joint_names, '压力值': pressures})
    return fig, table_data.to_dict('records')


@app.callback(
    [Output('hand-visualization-graph-left', 'figure'),
     Output('hand-visualization-graph-right', 'figure'),
     Output('current-frame-table-left', 'data'),
     Output('current-frame-table-right', 'data')],
    [Input('timeline-slider', 'value'),
     Input('manual-max-pressure-input', 'value')],
    [State('processed-data-store-left', 'data'),
     State('processed-data-store-right', 'data'),
     State('pressure-range-store', 'data')]
)
def update_file_visualizations(frame_index, manual_max, json_data_l, json_data_r, range_data):
    df_l = pd.read_json(json_data_l, orient='split') if json_data_l else None
    df_r = pd.read_json(json_data_r, orient='split') if json_data_r else None

    fig_l, table_l = create_figure_from_file(df_l, frame_index, range_data, manual_max, 'left')
    fig_r, table_r = create_figure_from_file(df_r, frame_index, range_data, manual_max, 'right')
    
    return fig_l, fig_r, table_l, table_r

@app.callback(
    [Output('hand-visualization-graph-realtime', 'figure'),
     Output('current-frame-table-realtime', 'data')],
    [Input('realtime-interval', 'n_intervals')],
    [State('manual-max-pressure-input', 'value'),
     State('pressure-range-store', 'data')]
)
def update_realtime_visualization(n, manual_max, range_data):
    global current_realtime_frame
    if not is_reading:
        return no_update, no_update
    fig, table = create_figure_realtime(current_realtime_frame, range_data, manual_max)
    return fig, table
# endregion

# region 实时数据采集控制
@app.callback(
    [Output('status-message', 'children'),
     Output('realtime-interval', 'disabled')],
    [Input('connect-button', 'n_clicks'),
     Input('disconnect-button', 'n_clicks')],
    [State('port-dropdown', 'value')],
    prevent_initial_call=True
)
def update_connection(connect_clicks, disconnect_clicks, port):
    global ser, serial_thread, is_reading, last_frame_time
    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    status_message = "未连接"
    interval_disabled = True

    if button_id == 'connect-button' and port:
        if not (ser and ser.is_open):
            try:
                ser = serial.Serial(port, 115200, timeout=0.1)
                is_reading = True
                last_frame_time = 0 
                serial_thread = threading.Thread(target=read_from_port, daemon=True)
                serial_thread.start()
                status_message = f"已连接到 {port}"
                interval_disabled = False
            except serial.SerialException as e:
                ser, is_reading = None, False
                status_message = f"连接失败: {e}"

    elif button_id == 'disconnect-button':
        if ser and ser.is_open:
            is_reading = False
            if serial_thread and serial_thread.is_alive():
                serial_thread.join(timeout=0.5) 
            ser.close()
            ser = None
        status_message = "已断开"
        
    return status_message, interval_disabled

@app.callback(
    Output('record-status', 'children'),
    [Input('start-record-button', 'n_clicks'),
     Input('stop-record-button', 'n_clicks')],
    prevent_initial_call=True
)
def update_recording(start_clicks, stop_clicks):
    global is_recording, recorded_data
    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    message = dash.no_update

    if button_id == 'start-record-button':
        if not is_recording:
            is_recording = True
            recorded_data = [] # 清空旧数据
            message = "🔴 正在记录..."
        else:
            message = "已在记录中"

    elif button_id == 'stop-record-button':
        if is_recording:
            is_recording = False
            message = f"记录已停止，共记录 {len(recorded_data)} 帧"
        else:
            message = "未在记录"
    
    return message
# endregion

# region 数据导出回调
@app.callback(
    Output("download-zip", "data"),
    Input("export-zip-button", "n_clicks"),
    [State('processed-data-store-left', 'data'),
     State('processed-data-store-right', 'data')],
    prevent_initial_call=True,
)
def export_data_as_zip(n_clicks, json_data_l, json_data_r):
    files = []
    if json_data_l:
        df_l = pd.read_json(json_data_l, orient='split')
        files.append(dict(name='left_hand_data.csv', data=df_l.to_csv(index=False).encode()))
    if json_data_r:
        df_r = pd.read_json(json_data_r, orient='split')
        files.append(dict(name='right_hand_data.csv', data=df_r.to_csv(index=False).encode()))
    
    if not files: return no_update

    zs = ZipStream(files)
    return Response(zs.stream(), mimetype='application/zip', headers={
        "Content-Disposition": "attachment; filename=tactile_data_files.zip"
    })

@app.callback(
    Output("download-realtime-zip", "data"),
    Input("export-realtime-button", "n_clicks"),
    prevent_initial_call=True,
)
def export_realtime_data_as_zip(n_clicks):
    global recorded_data
    if not recorded_data:
        return no_update

    # 获取 MANO 关键点名称作为表头
    header = ['timestamp'] + list(MANO_2D_COORDS_LEFT.keys())
    
    # 将数据转换为 CSV 格式的字符串
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(header)
    writer.writerows(recorded_data)
    csv_data = output.getvalue().encode('utf-8')
    
    files = [dict(name='realtime_tactile_data.csv', data=csv_data)]
    zs = ZipStream(files)

    # 导出后清空记录
    recorded_data = []
    
    return Response(zs.stream(), mimetype='application/zip', headers={
        "Content-Disposition": f"attachment; filename=realtime_tactile_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    })

# endregion

# --- 运行 App ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
