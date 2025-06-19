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

from tactile_processor import load_and_process_tactile_data

# --- 手部可视化设置 (无变动) ---
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

# --- App 初始化 ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
server = app.server
app.title = "双触觉数据可视化"

# --- 辅助函数 ---
def make_hand_panel(hand_type):
    """创建左手或右手的UI面板"""
    title = "左手 (Left Hand)" if hand_type == 'left' else "右手 (Right Hand)"
    return dbc.Card([
        dbc.CardHeader(title),
        dbc.CardBody([
            dcc.Upload(
                id=f'upload-data-{hand_type}',
                children=html.Div(['拖拽或 ', html.A('选择CSV文件')]),
                style={'width': '100%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'margin-bottom': '10px'},
            ),
            dcc.Graph(id=f'hand-visualization-graph-{hand_type}', style={'height': '50vh'}),
            dash_table.DataTable(id=f'current-frame-table-{hand_type}', style_cell={'textAlign': 'left'}, style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'})
        ])
    ])

# --- App 布局 ---
app.layout = dbc.Container([
    dcc.Store(id='processed-data-store-left'),
    dcc.Store(id='processed-data-store-right'),
    dcc.Store(id='play-state-store', data={'is_playing': False}),
    dcc.Store(id='pressure-range-store'),
    dcc.Interval(id='animation-interval', interval=200, n_intervals=0, disabled=True),
    dcc.Download(id="download-zip"),

    dbc.Row(dbc.Col(html.H1("双手触觉数据可视化"), width=12), className="mt-4 mb-2 text-center"),
    dbc.Row([
        dbc.Col(make_hand_panel('left'), md=5),
        dbc.Col(dbc.Card([
            dbc.CardHeader("同步控制器"),
            dbc.CardBody([
                html.P("时间线", className="font-weight-bold"),
                dcc.Slider(id='timeline-slider', min=0, max=0, value=0, step=1, marks=None, tooltip={"placement": "bottom", "always_visible": True}),
                html.Div([
                    dbc.Button("▶️ 播放", id="play-button", color="success", className="me-2"),
                    dbc.Button("⏸️ 暂停", id="pause-button", color="warning", className="me-2"),
                    dbc.Button("🔄 重置", id="reset-button", color="info"),
                ], className="mt-3 text-center"),
                html.Hr(),
                html.P("可视化颜色范围", className="font-weight-bold"),
                dbc.InputGroup([
                    dbc.InputGroupText("最大压力值"),
                    dbc.Input(id="manual-max-pressure-input", type="number", placeholder="自动"),
                ], className="mb-3"),
                html.Hr(),
                dbc.Button("导出为ZIP压缩包", id="export-zip-button", color="primary", className="w-100"),
            ])
        ]), md=2),
        dbc.Col(make_hand_panel('right'), md=5),
    ], className="mt-3"),
], fluid=True)


# --- 回调函数 (Callbacks) ---
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

# --- 函数定义中的关键修复 ---
def create_figure(df, frame_index, range_data, manual_max, hand_type):
    """为单只手创建可视化图表"""
    # 修复: 确保在任何情况下都返回两个值 (figure, table_data)
    if df is None or frame_index >= len(df) or range_data is None:
        fig = go.Figure(layout={'xaxis': {'visible': False}, 'yaxis': {'visible': False}, 'title': f'{hand_type.capitalize()} Hand - No Data'})
        return fig, [] # 返回一个空图和一个空列表

    coords = MANO_2D_COORDS_LEFT if hand_type == 'left' else MANO_2D_COORDS_RIGHT
    current_frame_data = df.iloc[frame_index]
    
    min_pressure = range_data.get('min', 0)
    max_pressure_for_color = manual_max if manual_max is not None and manual_max > min_pressure else range_data.get('p99', 1.0)
    pressure_range = max(1.0, max_pressure_for_color - min_pressure)

    points_x = [coords[j]['x'] for j in df.columns if j != 'timestamp']
    points_y = [coords[j]['y'] for j in df.columns if j != 'timestamp']
    pressures = [current_frame_data[j] for j in df.columns if j != 'timestamp']
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
# --- 修复结束 ---

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
def update_all_visualizations(frame_index, manual_max, json_data_l, json_data_r, range_data):
    df_l = pd.read_json(json_data_l, orient='split') if json_data_l else None
    df_r = pd.read_json(json_data_r, orient='split') if json_data_r else None

    fig_l, table_l = create_figure(df_l, frame_index, range_data, manual_max, 'left')
    fig_r, table_r = create_figure(df_r, frame_index, range_data, manual_max, 'right')
    
    return fig_l, fig_r, table_l, table_r

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
        "Content-Disposition": "attachment; filename=tactile_data.zip"
    })

# --- 运行 App ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)