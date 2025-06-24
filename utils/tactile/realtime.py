# tactile_sensor_app.py

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import numpy as np
import serial
import serial.tools.list_ports
import threading
import time
from collections import deque
import pandas as pd
from datetime import datetime
import base64
import io
import xml.etree.ElementTree as ET
from xml.dom import minidom
import csv

# =============================================================================
# 全局变量和配置
# =============================================================================
# 传感器尺寸
SENSOR_ROWS = 16
SENSOR_COLS = 16
NUM_POINTS = SENSOR_ROWS * SENSOR_COLS
PACKET_START_CHAR = 0x24 # '$'
PACKET_SIZE = 10 # 1 byte start, 1 byte ID, 8 bytes payload

# 数据存储
data_queue = deque(maxlen=100)
current_raw_frame = np.zeros((SENSOR_ROWS, SENSOR_COLS))
current_calibrated_frame = np.zeros((SENSOR_ROWS, SENSOR_COLS))
last_frame_time = 0

# 校准参数 (K: 增益, B: 偏移)
calibration_params = pd.DataFrame({
    'K': [1.0] * NUM_POINTS,
    'B': [0.0] * NUM_POINTS
})

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
csv_writer = None
csv_file = None

# =============================================================================
# 数据处理和串口通信 (v1.2 核心更新)
# =============================================================================
def apply_calibration(raw_frame):
    """应用线性校准 Y = K*X + B"""
    raw_flat = raw_frame.flatten()
    k_vals = calibration_params['K'].values
    b_vals = calibration_params['B'].values
    calibrated_flat = k_vals * raw_flat + b_vals
    return calibrated_flat.reshape((SENSOR_ROWS, SENSOR_COLS))

def read_from_port():
    """
    在单独的线程中从串口读取和解析数据。
    v1.2更新：完全仿照原始C#软件的ASCII数据包处理逻辑。
    """
    global ser, is_reading, current_raw_frame, current_calibrated_frame, data_queue, last_frame_time
    
    buffer = bytearray()
    # 用于重组一帧完整数据的临时数组
    data_accumulator = np.zeros(NUM_POINTS, dtype=int)

    while is_reading and ser and ser.is_open:
        try:
            # 1. 读取所有可用数据并追加到缓冲区
            if ser.in_waiting > 0:
                buffer.extend(ser.read(ser.in_waiting))
            
            # 2. 只要缓冲区数据量足够一个包，就持续尝试处理
            while len(buffer) >= PACKET_SIZE:
                # 3. 严格模仿C#逻辑：只检查缓冲区头部是否为'$'
                if buffer[0] == PACKET_START_CHAR:
                    # 获取包ID
                    packet_id = buffer[1]
                    
                    # 验证包ID (必须是8的倍数，且小于等于24)
                    if packet_id <= 24 and packet_id % 8 == 0:
                        # 提取并解码8个字节的数据负载
                        for i in range(8):
                            # 解码: 原始值 = 接收值 - 40
                            decoded_value = buffer[2 + i] - 40
                            data_accumulator[packet_id + i] = decoded_value

                        # 如果这是最后一个包(ID=24)，则意味着一帧数据已完整
                        if packet_id == 24:
                            # 更新全局数据
                            last_frame_time = time.time()
                            # 将重组后的数据变形为16x16的矩阵
                            raw_frame = data_accumulator.reshape((SENSOR_ROWS, SENSOR_COLS))
                            current_raw_frame = raw_frame
                            current_calibrated_frame = apply_calibration(raw_frame)
                            
                            max_val = np.max(current_calibrated_frame)
                            data_queue.append({'time': last_frame_time, 'max_value': max_val})

                            # 数据记录
                            if is_recording and csv_writer is not None:
                                flat_raw = current_raw_frame.flatten()
                                flat_calibrated = current_calibrated_frame.flatten()
                                row = [datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]] + list(flat_raw) + list(flat_calibrated)
                                csv_writer.writerow(row)
                        
                        # 清理缓冲区：移除已处理的包
                        del buffer[:PACKET_SIZE]
                    else:
                        # 包ID无效，丢弃'$'符号，继续寻找下一个
                        del buffer[0]
                else:
                    # 头部不是'$'，丢弃第一个无效字节
                    del buffer[0]
            
            # 当缓冲区数据不足一个包时，稍作等待
            time.sleep(0.01)

        except serial.SerialException as se:
            print(f"串口错误: {se}")
            is_reading = False
            break
        except Exception as e:
            print(f"读取或解析数据时出错: {e}")
            buffer.clear()
            time.sleep(0.01)

# =============================================================================
# Dash 应用
# =============================================================================
app = dash.Dash(__name__, title="触觉传感器操作软件 v1.2", suppress_callback_exceptions=True)
server = app.server

# -----------------------------------------------------------------------------
# 应用布局 (与之前版本相同)
# -----------------------------------------------------------------------------
app.layout = html.Div(style={'fontFamily': 'Arial, sans-serif', 'padding': '20px'}, children=[
    html.H1("触觉传感器操作软件 v1.2 (最终修复版)", style={'textAlign': 'center', 'color': '#003366'}),
    html.Hr(),

    html.Div(className='control-panel', style={'marginBottom': '20px', 'padding': '15px', 'border': '1px solid #ddd', 'borderRadius': '5px'}, children=[
        html.H4("通信与记录", style={'marginTop': '0'}),
        html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '15px'}, children=[
            dcc.Dropdown(id='port-dropdown', options=[{'label': p, 'value': p} for p in available_ports], placeholder="选择串口", style={'width': '150px'}),
            dcc.Input(id='baudrate-input', type='number', value=115200, placeholder="波特率", style={'width': '120px'}),
            html.Button('连接', id='connect-button', n_clicks=0, style={'backgroundColor': '#4CAF50', 'color': 'white', 'border': 'none', 'padding': '8px 16px', 'borderRadius': '5px'}),
            html.Button('断开', id='disconnect-button', n_clicks=0, style={'backgroundColor': '#f44336', 'color': 'white', 'border': 'none', 'padding': '8px 16px', 'borderRadius': '5px'}),
            html.Div(id='connection-status', style={'width': '20px', 'height': '20px', 'borderRadius': '50%', 'backgroundColor': 'grey'}),
            html.Div(id='status-message', style={'flexGrow': 1}),
        ]),
        html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '15px', 'marginTop': '15px'}, children=[
            html.Button('开始记录', id='start-record-button', n_clicks=0, style={'backgroundColor': '#008CBA', 'color': 'white', 'border': 'none', 'padding': '8px 16px', 'borderRadius': '5px'}),
            html.Button('停止记录', id='stop-record-button', n_clicks=0, style={'backgroundColor': '#e7e7e7', 'color': 'black', 'border': 'none', 'padding': '8px 16px', 'borderRadius': '5px'}),
            html.Div(id='record-status', style={'color': 'grey'}),
        ]),
    ]),
    
    dcc.Tabs(id="tabs-main", value='tab-main-interface', children=[
        dcc.Tab(label='主界面', value='tab-main-interface', children=[
            html.Div(style={'display': 'grid', 'gridTemplateColumns': '2fr 1fr', 'gap': '20px', 'paddingTop': '20px'}, children=[
                html.Div(children=[
                    html.Div(className='heatmap-container', style={'border': '1px solid #ddd', 'borderRadius': '5px', 'padding': '15px'}, children=[
                        html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center'}, children=[
                            html.H4("实时压力分布", style={'marginTop': '0', 'marginBottom': '10px'}),
                            dcc.RadioItems(
                                id='view-mode-selector',
                                options=[
                                    {'label': '校准后数据', 'value': 'calibrated'},
                                    {'label': '原始数据', 'value': 'raw'},
                                ],
                                value='calibrated',
                                inline=True,
                                labelStyle={'margin-right': '15px'}
                            ),
                        ]),
                        dcc.Graph(id='heatmap-graph'),
                    ]),
                    html.Div(className='linechart-container', style={'marginTop': '20px', 'border': '1px solid #ddd', 'borderRadius': '5px', 'padding': '15px'}, children=[
                        html.H4("实时数据曲线 (最大值)", style={'marginTop': '0'}),
                        dcc.Graph(id='line-chart-graph'),
                    ]),
                ]),
                html.Div(className='overview-container', style={'border': '1px solid #ddd', 'borderRadius': '5px', 'padding': '15px'}, children=[
                    html.H4(id='overview-title', style={'marginTop': '0'}),
                    html.Div(id='overview-display', style={'fontSize': '1.1em', 'lineHeight': '1.8'}),
                ]),
            ]),
        ]),
        
        dcc.Tab(label='参数管理', value='tab-param-mgmt', children=[
            html.Div(style={'padding': '20px'}, children=[
                html.H4("校准参数 (K: 增益, B: 偏移)"),
                html.Div(style={'display': 'flex', 'gap': '15px', 'marginBottom': '15px'}, children=[
                    dcc.Upload(id='upload-cal-file', children=html.Button('加载校准文件 (XML)'), multiple=False),
                    html.Button('保存校准文件 (XML)', id='save-cal-button', n_clicks=0),
                    html.Button('应用表格中的修改', id='apply-param-button', n_clicks=0),
                ]),
                html.Div(id='param-status', style={'marginBottom': '15px', 'color': 'blue'}),
                dcc.Download(id="download-xml"),
                dash_table.DataTable(
                    id='param-table',
                    columns=[{"name": i, "id": i} for i in ['Channel', 'K', 'B']],
                    data=[{'Channel': i, 'K': 1.0, 'B': 0.0} for i in range(NUM_POINTS)],
                    editable=True, page_size=20, style_cell={'textAlign': 'left'}, style_header={'fontWeight': 'bold'},
                ),
            ]),
        ]),
        
        dcc.Tab(label='线性拟合', value='tab-linear-fit', children=[
            html.Div(style={'padding': '20px', 'display': 'grid', 'gridTemplateColumns': '1fr 2fr', 'gap': '20px'}, children=[
                 html.Div([
                    html.H4("三点线性校准"),
                    html.P("说明：在此对单个通道进行校准。先施加载荷，待下方读数稳定后，点击对应的“获取”按钮锁定读数。完成至少两个点的载荷和读数后，点击“开始校准”即可。"),
                    html.Label("选择校准通道 (0-255):"),
                    dcc.Input(id='cal-channel-input', type='number', value=0, min=0, max=NUM_POINTS-1, style={'width': '100%', 'marginBottom': '10px'}),
                    html.Hr(),
                    html.Label("当前通道原始读数:"),
                    html.Div(id='cal-raw-reading', style={'fontWeight': 'bold', 'fontSize': '1.5em', 'color': '#007BFF', 'marginBottom': '20px'}),
                    html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '5px', 'marginBottom': '5px'}, children=[
                        html.Label("点1 - 载荷(g):", style={'width': '100px'}),
                        dcc.Input(id='load-1', type='number', placeholder='例: 500', style={'flex': 1}),
                        html.Label("读数:", style={'marginLeft': '10px'}),
                        dcc.Input(id='reading-1', type='number', style={'flex': 1}, disabled=True),
                        html.Button('获取', id='get-reading-1', n_clicks=0),
                    ]),
                     html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '5px', 'marginBottom': '5px'}, children=[
                        html.Label("点2 - 载荷(g):", style={'width': '100px'}),
                        dcc.Input(id='load-2', type='number', placeholder='例: 1000', style={'flex': 1}),
                        html.Label("读数:", style={'marginLeft': '10px'}),
                        dcc.Input(id='reading-2', type='number', style={'flex': 1}, disabled=True),
                        html.Button('获取', id='get-reading-2', n_clicks=0),
                    ]),
                     html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '5px', 'marginBottom': '5px'}, children=[
                        html.Label("点3 - 载荷(g):", style={'width': '100px'}),
                        dcc.Input(id='load-3', type='number', placeholder='例: 2000', style={'flex': 1}),
                        html.Label("读数:", style={'marginLeft': '10px'}),
                        dcc.Input(id='reading-3', type='number', style={'flex': 1}, disabled=True),
                        html.Button('获取', id='get-reading-3', n_clicks=0),
                    ]),
                    html.Button('开始校准', id='start-cal-button', n_clicks=0, style={'width': '100%', 'marginTop': '20px', 'padding': '10px'}),
                    html.Div(id='cal-status', style={'marginTop': '10px', 'color': 'red'}),
                ]),
                html.Div([
                    dcc.Graph(id='cal-graph'),
                    html.Div(id='cal-results', style={'fontWeight': 'bold', 'fontSize': '1.2em'}),
                ]),
            ]),
        ]),
    ]),
    
    dcc.Interval(id='interval-component', interval=100, n_intervals=0),
])

# -----------------------------------------------------------------------------
# 回调函数
# -----------------------------------------------------------------------------
@app.callback(
    [Output('connection-status', 'style'), Output('status-message', 'children')],
    [Input('connect-button', 'n_clicks'), Input('disconnect-button', 'n_clicks')],
    [State('port-dropdown', 'value'), State('baudrate-input', 'value')]
)
def update_connection(connect_clicks, disconnect_clicks, port, baudrate):
    global ser, serial_thread, is_reading, last_frame_time
    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'No clicks yet'
    
    status_style = {'width': '20px', 'height': '20px', 'borderRadius': '50%', 'backgroundColor': 'grey'}
    status_message = "未连接"

    if button_id == 'connect-button' and port:
        if not (ser and ser.is_open):
            try:
                ser = serial.Serial(port, baudrate, timeout=0.1)
                is_reading = True
                last_frame_time = 0 
                serial_thread = threading.Thread(target=read_from_port, daemon=True)
                serial_thread.start()
            except serial.SerialException as e:
                ser, is_reading = None, False
                status_style['backgroundColor'] = 'red'
                status_message = f"连接失败: {e}"

    elif button_id == 'disconnect-button':
        if ser and ser.is_open:
            is_reading = False
            if serial_thread and serial_thread.is_alive():
                serial_thread.join(timeout=0.5) 
            ser.close()
            ser = None
        
    if ser and ser.is_open:
        status_style['backgroundColor'] = 'green'
        status_message = f"已连接到 {ser.port} @ {ser.baudrate} bps"
        
    return status_style, status_message

@app.callback(
    Output('record-status', 'children'),
    [Input('start-record-button', 'n_clicks'), Input('stop-record-button', 'n_clicks')]
)
def update_recording(start_clicks, stop_clicks):
    global is_recording, csv_writer, csv_file
    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'No clicks yet'
    message = dash.no_update

    if button_id == 'start-record-button' and not is_recording:
        is_recording = True
        filename = f"data_record_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        csv_file = open(filename, 'w', newline='', encoding='utf-8-sig')
        csv_writer = csv.writer(csv_file)
        
        raw_headers = [f'raw_{i}' for i in range(NUM_POINTS)]
        cal_headers = [f'cal_{i}' for i in range(NUM_POINTS)]
        header = ['timestamp'] + raw_headers + cal_headers
        csv_writer.writerow(header)
        message = f"正在记录数据到 {filename}"

    elif button_id == 'stop-record-button' and is_recording:
        is_recording = False
        if csv_file:
            csv_file.close()
            csv_file, csv_writer = None, None
        message = "记录已停止"
    
    return message

# --- 主界面 ---
@app.callback(
    Output('heatmap-graph', 'figure'),
    [Input('interval-component', 'n_intervals'), Input('view-mode-selector', 'value')]
)
def update_heatmap(n, view_mode):
    data_to_show = current_raw_frame if view_mode == 'raw' else current_calibrated_frame
    title = '原始数据' if view_mode == 'raw' else '校准后数据'
        
    fig = go.Figure(data=go.Heatmap(z=data_to_show, colorscale='Viridis'))
    fig.update_layout(
        title=title,
        margin=dict(l=20, r=20, t=40, b=20), 
        yaxis=dict(scaleanchor="x", scaleratio=1, autorange='reversed')
    )
    return fig

@app.callback(
    [Output('overview-display', 'children'), Output('overview-title', 'children')],
    [Input('interval-component', 'n_intervals'), Input('view-mode-selector', 'value')]
)
def update_overview(n, view_mode):
    if not (ser and ser.is_open and is_reading):
        return "未连接", "数据总览"

    if last_frame_time == 0:
        return "正在等待第一帧数据...", "数据总览"

    if time.time() - last_frame_time > 2:
        return "数据流中断，请检查传感器连接。", "数据总览"

    data_to_show = current_raw_frame if view_mode == 'raw' else current_calibrated_frame
    title = f"数据总览 ({'原始值' if view_mode == 'raw' else '校准后'})"
    
    max_val, min_val, sum_val, avg_val = np.max(data_to_show), np.min(data_to_show), np.sum(data_to_show), np.mean(data_to_show)
    max_coords = np.unravel_index(np.argmax(data_to_show), data_to_show.shape)
    
    overview_content = [
        html.P(f"最大值: {max_val:.2f}"),
        html.P(f"最大值坐标: ({max_coords[0]}, {max_coords[1]})"),
        html.P(f"最小值: {min_val:.2f}"),
        html.P(f"平均值: {avg_val:.2f}"),
        html.P(f"总和: {sum_val:.2f}")
    ]
    return overview_content, title

@app.callback(
    Output('line-chart-graph', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_line_chart(n):
    times = [d['time'] for d in data_queue]
    max_values = [d['max_value'] for d in data_queue]
    fig = go.Figure(data=go.Scatter(x=list(times), y=list(max_values), mode='lines'))
    fig.update_layout(margin=dict(l=20, r=20, t=30, b=20), xaxis_title="时间", yaxis_title="压力值 (校准后)")
    return fig

# --- 参数管理 ---
@app.callback(
    [Output('param-table', 'data'), Output('param-status', 'children')],
    [Input('upload-cal-file', 'contents'), Input('apply-param-button', 'n_clicks')],
    [State('upload-cal-file', 'filename'), State('param-table', 'data')]
)
def update_parameters(contents, apply_clicks, filename, table_data):
    global calibration_params
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'No clicks yet'
    
    msg = dash.no_update
    if trigger_id == 'upload-cal-file' and contents is not None:
        _, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            tree = ET.parse(io.StringIO(decoded.decode('utf-8')))
            root = tree.getroot()
            temp_k, temp_b = [], []
            for i in range(NUM_POINTS):
                k_val = float(root.find(f'K_{i}').text) if root.find(f'K_{i}') is not None else 1.0
                b_val = float(root.find(f'B_{i}').text) if root.find(f'B_{i}') is not None else 0.0
                temp_k.append(k_val)
                temp_b.append(b_val)
            
            calibration_params['K'], calibration_params['B'] = temp_k, temp_b
            msg = f"成功加载校准文件: {filename}"
        except Exception as e:
            msg = f"加载文件失败: {e}"

    elif trigger_id == 'apply-param-button':
        df = pd.DataFrame(table_data)
        calibration_params['K'] = df['K'].astype(float).values
        calibration_params['B'] = df['B'].astype(float).values
        msg = f"表格中的参数已于 {datetime.now().strftime('%H:%M:%S')} 应用"

    new_table_data = [{'Channel': i, 'K': calibration_params.loc[i, 'K'], 'B': calibration_params.loc[i, 'B']} for i in range(NUM_POINTS)]
    return new_table_data, msg

@app.callback(
    Output("download-xml", "data"),
    Input("save-cal-button", "n_clicks"),
    State("param-table", "data"),
    prevent_initial_call=True,
)
def save_parameters_to_xml(n_clicks, table_data):
    root = ET.Element('root')
    for row in table_data:
        channel, k_val, b_val = row['Channel'], row['K'], row['B']
        ET.SubElement(root, f'K_{channel}').text = str(k_val)
        ET.SubElement(root, f'B_{channel}').text = str(b_val)

    xml_str = ET.tostring(root, 'utf-8')
    pretty_xml_str = minidom.parseString(xml_str).toprettyxml(indent="   ")
    
    return dict(content=pretty_xml_str, filename=f"Config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xml")

# --- 线性拟合 ---
@app.callback(
    Output('cal-raw-reading', 'children'),
    [Input('interval-component', 'n_intervals'), Input('cal-channel-input', 'value')]
)
def update_cal_raw_reading(n, channel):
    if channel is not None and 0 <= channel < NUM_POINTS:
        row, col = divmod(channel, SENSOR_COLS)
        return f"{current_raw_frame[row, col]}"
    return "N/A"

@app.callback(
    [Output('reading-1', 'value'), Output('reading-2', 'value'), Output('reading-3', 'value')],
    [Input('get-reading-1', 'n_clicks'), Input('get-reading-2', 'n_clicks'), Input('get-reading-3', 'n_clicks'), Input('cal-channel-input', 'value')],
    [State('cal-raw-reading', 'children'), State('reading-1', 'value'), State('reading-2', 'value'), State('reading-3', 'value')],
    prevent_initial_call=True,
)
def get_readings(n1, n2, n3, channel, raw_reading, r1, r2, r3):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if not raw_reading or raw_reading == 'N/A':
        return r1, r2, r3

    if trigger_id == 'get-reading-1': r1 = int(raw_reading)
    elif trigger_id == 'get-reading-2': r2 = int(raw_reading)
    elif trigger_id == 'get-reading-3': r3 = int(raw_reading)
    elif 'cal-channel-input' in trigger_id: return None, None, None
        
    return r1, r2, r3

@app.callback(
    [Output('cal-graph', 'figure'), Output('cal-results', 'children'), Output('cal-status', 'children')],
    [Input('start-cal-button', 'n_clicks')],
    [State('load-1', 'value'), State('reading-1', 'value'),
     State('load-2', 'value'), State('reading-2', 'value'),
     State('load-3', 'value'), State('reading-3', 'value'),
     State('cal-channel-input', 'value')],
    prevent_initial_call=True,
)
def perform_calibration_fit(n_clicks, l1, r1, l2, r2, l3, r3, channel):
    loads, readings = [l1, l2, l3], [r1, r2, r3]
    valid_points = [(r, l) for r, l in zip(readings, loads) if r is not None and l is not None]

    if len(valid_points) < 2:
        return go.Figure(), "", "错误：至少需要两个有效的校准点。"

    readings_valid, loads_valid = np.array([p[0] for p in valid_points]), np.array([p[1] for p in valid_points])

    try:
        k, b = np.polyfit(readings_valid, loads_valid, 1)
        if np.all(readings_valid == readings_valid[0]):
             r_squared = 1.0 if np.all(loads_valid == loads_valid[0]) else 0.0
        else:
            r_squared = np.corrcoef(readings_valid, loads_valid)[0, 1]**2

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=readings_valid, y=loads_valid, mode='markers', name='校准点', marker=dict(size=10)))
        fit_x = np.linspace(min(readings_valid) - 5, max(readings_valid) + 5, 10)
        fig.add_trace(go.Scatter(x=fit_x, y=k * fit_x + b, mode='lines', name='拟合线'))
        fig.update_layout(title=f"通道 {channel} 的线性拟合结果", xaxis_title="软件原始示数", yaxis_title="施加载荷 (g)")
        
        results_text = f"增益 K: {k:.4f} | 偏移 B: {b:.2f} | 相关系数 R²: {r_squared:.4f}"
        
        global calibration_params
        calibration_params.loc[channel, 'K'] = k
        calibration_params.loc[channel, 'B'] = b
        
        return fig, results_text, f"校准完成并已应用到通道 {channel}！"
    except Exception as e:
        return go.Figure(), "", f"计算错误: {e}"

# =============================================================================
# 运行应用
# =============================================================================
if __name__ == '__main__':
    app.run(debug=True)
