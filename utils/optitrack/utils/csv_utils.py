import pandas as pd
import torch
from pytorch3d.transforms import quaternion_to_axis_angle
import numpy as np

# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


def read_optitrack_data(filepath):
    df = pd.read_csv(filepath, skiprows=2, low_memory=False)

    # split base on “Bone” and “Rigid Body”
    df_bone = df.filter(like="Bone")
    df_rigid_body = df.filter(like="Rigid Body")

    # print(set(x[8:] for x in df_bone.iloc[0]))
    # save the human and obj set
    human_list = list(set(x.split(":")[0] for x in df_bone.iloc[0]))
    obj_list = list(set(x.split(":")[0] for x in df_rigid_body.iloc[0]))

    # deal with Bone data
    df_bone.columns = [
        f"{name}:{type_}:{index}"
        for name, type_, index in zip(
            df_bone.iloc[0], df_bone.iloc[2], df_bone.iloc[3]
        )
    ]
    df_bone = df_bone.drop([0, 1, 2, 3]).reset_index(drop=True).astype(float)

    # deal with Bone data
    df_rigid_body.columns = [
        f"{name}:{type_}:{index}"
        for name, type_, index in zip(
            df_rigid_body.iloc[0], df_rigid_body.iloc[2], df_rigid_body.iloc[3]
        )
    ]
    df_rigid_body = (
        df_rigid_body.drop([0, 1, 2, 3]).reset_index(drop=True).astype(float)
    )

    return df_bone, df_rigid_body, human_list, obj_list


# def get_joint_axis_angle(df_bone, joint, human_name):
#     col_names = [
#         ":".join([human_name, joint, "Rotation", quat])
#         for quat in ("W", "X", "Y", "Z")
#     ]
#     quat = torch.tensor(
#         df_bone[col_names].values, dtype=torch.float32, device=device
#     )  # (N, 4)
#     return quaternion_to_axis_angle(quat)
# Fix "ValueError: Found zero norm quaternions in 'quat'" when Optitrack loses marker tracking
def get_joint_axis_angle(df_bone, joint, human_name):
    col_names = [
        ":".join([human_name, joint, "Rotation", quat])
        for quat in ("W", "X", "Y", "Z")
    ]
    # Load data, ensuring that any potential NaN values from pandas are converted to 0.0
    values = df_bone[col_names].fillna(0.0).values
    quat = torch.tensor(
        values, dtype=torch.float32, device=device
    )  # (N, 4)

    # Find quaternions with a norm close to zero, which are invalid.
    norms = torch.linalg.norm(quat, dim=1)
    zero_norm_indices = torch.where(norms < 1e-6)[0]

    # Replace these invalid quaternions with an identity quaternion [w=1, x=0, y=0, z=0],
    # which represents no rotation.
    if len(zero_norm_indices) > 0:
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device)
        quat[zero_norm_indices] = identity_quat
        # print(f"INFO: Found and fixed {len(zero_norm_indices)} invalid zero-norm quaternions for joint '{joint}'.")

    return quaternion_to_axis_angle(quat)


def get_quaternion_manus(df_bone, joint):
    col_names = [
        "_".join([joint, quat]) for quat in ("W", "X.1", "Y.1", "Z.1")
    ]
    quat = torch.tensor(
        df_bone[col_names].values, dtype=torch.float32, device=device
    )  # (N, 4)
    return quat


def get_quaternion_inverse(quat):
    """返回四元数的共轭，即取反虚部，单位四元数共轭与取逆等价"""
    return torch.cat((quat[:, :1], -quat[:, 1:]), dim=1)

def quaternion_multiply(q1, q2):
    """计算两个四元数的乘积"""
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    
    return torch.stack((
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ), dim=1)


def read_manus_data(filepath):
    df = pd.read_csv(filepath)
    
    # choose the available columns
    parts = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
    Joints = ["CMC", "MCP", "PIP", "DIP", "TIP"]
    columns = ["X.1", "Y.1", "Z.1", "W"]

    available_columns = []
    for part in parts:
        for joint in Joints:
            if part == "Thumb" and joint == "PIP":
                continue
            for column in columns:
                available_columns.append(f"{part}_{joint}_{column}")
    available_columns += ["Hand_X.1", "Hand_Y.1", "Hand_Z.1", "Hand_W"]

    df_bone = df[available_columns]

    return df_bone

def get_joint_position(df_bone, joint, human_name):
    """
    Extracts the 3D position of a specific joint over time.
    """
    col_names = [
        ":".join([human_name, joint, "Position", pos])
        for pos in ("X", "Y", "Z")
    ]
    position_df = df_bone[col_names]

    # Use pandas' interpolate method to fill NaN values.
    # 'linear' interpolation creates a smooth transition for missing frames.
    # 'limit_direction="both"' ensures that NaNs at the very start or end are also filled.
    interpolated_df = position_df.interpolate(method='linear', limit_direction='both')

    # Convert the cleaned DataFrame's values to a tensor.
    # Note: OptiTrack may export position in millimeters.
    # If so, you might need to divide the values by 1000 to convert to meters.
    pos = torch.tensor(
        interpolated_df.values, dtype=torch.float32, device=device
    )
    
    return pos