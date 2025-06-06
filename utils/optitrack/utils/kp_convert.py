from utils.csv_utils import get_joint_axis_angle
from utils.csv_utils import get_quaternion_manus
from utils.csv_utils import get_quaternion_inverse, quaternion_multiply
from pytorch3d.transforms import quaternion_to_axis_angle, axis_angle_to_quaternion
from utils.kp_convert_constants import *
from scipy.spatial.transform import Rotation as R
import torch
import math
import numpy as np


# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


def get_variable_value(var_name):
    # 检查全局变量字典中是否存在名为 var_name 的变量
    if var_name in globals():
        return globals()[var_name]
    else:
        raise NameError(f"Variable '{var_name}' is not defined")


def split_dict_into_batches(data_dict, batch_size):
    # 获取固定的 N 值
    N = data_dict["global_orient"].size(0)
    # 计算总的批次数
    num_batches = N // batch_size + (N % batch_size > 0)
    # 创建一个空列表来存储拆分后的字典
    batches = [{} for _ in range(num_batches)]
    # 遍历字典中的每个键值对
    for key, value in data_dict.items():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, N)
            # 将切片放入相应的字典中
            batches[i][key] = value[start_idx:end_idx]
    return batches


def get_body_pose(df_bone, model_type, human_name, batch=None):
    kpDict = get_variable_value("OPTI2" + model_type.upper())

    input_args = {}
    if model_type == "smpl":
        input_args = {
            "body_pose": torch.zeros(df_bone.shape[0], 23 * 3, device=device)
        }
        # prepare smpl input_args
        for joint, idx in kpDict.items():
            axis_angle = get_joint_axis_angle(df_bone, joint, human_name)
            if idx == 0:
                input_args["global_orient"] = axis_angle
            else:
                input_args["body_pose"][:, 3 * idx - 3 : 3 * idx] = axis_angle

    elif model_type == "smplh":
        input_args = {
            "body_pose": torch.zeros(
                1, model.NUM_BODY_JOINTS * 3, device=device
            ),
            "left_hand_pose": torch.zeros(
                1, model.NUM_HAND_JOINTS * 3, device=device
            ),
            "right_hand_pose": torch.zeros(
                1, model.NUM_HAND_JOINTS * 3, device=device
            ),
        }
        # TODO: prepare smplj input_args
    elif model_type == "smplx":
        input_args = {
            "body_pose": torch.zeros(
                1, model.NUM_BODY_JOINTS * 3, device=device
            ),
            "left_hand_pose": torch.zeros(
                1, model.NUM_HAND_JOINTS * 3, device=device
            ),
            "right_hand_pose": torch.zeros(
                1, model.NUM_HAND_JOINTS * 3, device=device
            ),
            "jaw_pose": torch.zeros(1, 3, device=device),
            "leye_pose": torch.zeros(1, 3, device=device),
            "reye_pose": torch.zeros(1, 3, device=device),
        }
        # TODO: prepare smplx input_args
    elif model_type == "mano":
        input_args = {
            "hand_pose": torch.zeros(
                1, model.NUM_HAND_JOINTS * 3, device=device
            )
        }
        # TODO: prepare mano input_args
    elif model_type == "flame":
        input_args = {
            "expression": torch.zeros(1, 10, device=device),
            "jaw_pose": torch.zeros(1, 3, device=device),
            "neck_pose": torch.zeros(1, 3, device=device),
            "leye_pose": torch.zeros(1, 3, device=device),
            "reye_pose": torch.zeros(1, 3, device=device),
        }
        # TODO: prepare flame input_args

    return split_dict_into_batches(input_args, batch) if batch else input_args

def get_parent_joint_idx(idx):
    parent_map = {
        0: None,   # Hand (手腕) 没有父节点
        13: 0,     # Thumb_CMC -> Hand
        14: 13,    # Thumb_MCP -> Thumb_CMC
        15: 14,    # Thumb_DIP -> Thumb_MCP
        1: 0,      # Index_MCP -> Hand
        2: 1,      # Index_PIP -> Index_MCP
        3: 2,      # Index_DIP -> Index_PIP
        4: 0,      # Middle_MCP -> Hand
        5: 4,      # Middle_PIP -> Middle_MCP
        6: 5,      # Middle_DIP -> Middle_PIP
        10: 0,     # Ring_MCP -> Hand
        11: 10,    # Ring_PIP -> Ring_MCP
        12: 11,    # Ring_DIP -> Ring_PIP
        7: 0,      # Pinky_MCP -> Hand
        8: 7,      # Pinky_PIP -> Pinky_MCP
        9: 8,      # Pinky_DIP -> Pinky_PIP
    }
    
    return parent_map.get(idx, None)  

def get_manus_pose(df_bone, batch=None):
    kpDict = get_variable_value("MANUS2MANO")

    input_args = {
        "hand_pose": torch.zeros(df_bone.shape[0], 15 * 3, device=device)
    }
    store_global_quat = {
        "hand_pose": torch.zeros(df_bone.shape[0], 15 * 4, device=device)
    }

    # prepare mano input_args
    for joint, idx in kpDict.items():
        quat = get_quaternion_manus(df_bone, joint)
        if idx == 0:
            store_global_quat["global_orient"] = quat
        else:
            parent_joint_idx = get_parent_joint_idx(idx)
            if parent_joint_idx != 0:
                parent_quat = store_global_quat["hand_pose"][:, 4 * parent_joint_idx - 4 : 4 * parent_joint_idx]
            else:
                parent_quat = store_global_quat["global_orient"]
         
            relative_quat = quaternion_multiply(get_quaternion_inverse(parent_quat), quat)
            store_global_quat["hand_pose"][:, 4 * idx - 4 : 4 * idx] = quat
            
            # relative_quat是子关节相对父关节在Manus全局参考系中的表达，需将其转换到父关节的坐标系中

            # input_args["hand_pose"][:, 3 * idx - 3 : 3 * idx] = quaternion_to_axis_angle(relative_quat)[:, [1, 0, 2]]
            
            # elif joint.startswith("Thumb_MCP"):
            #     result = quaternion_to_axis_angle(relative_quat)[:, [1, 2, 0]]
            #     # result[:, 1] *= -1
            # #     # result[:, 0] *= -1
            # #     input_args["hand_pose"][:, 3 * idx - 3 : 3 * idx] = result
            # # elif joint.startswith("Thumb_DIP"):
            # #     input_args["hand_pose"][:, 3 * idx - 3 : 3 * idx] = quaternion_to_axis_angle(relative_quat)[:, [1, 0, 2]]
            # if joint.startswith("Thumb_CMC"):
            #     thumb_coordinate = torch.from_numpy(np.load("thumb_CMC_coordinate.npy")).to(device)
            #     thumb_coordinate = thumb_coordinate.repeat(df_bone.shape[0], 1)
            #     thumb_rot_global = axis_angle_to_quaternion(quaternion_to_axis_angle(relative_quat)[:, [1, 2, 0]])
            #     thumb_input = quaternion_multiply(get_quaternion_inverse(thumb_coordinate), thumb_rot_global)
            #     input_args["hand_pose"][:, 3 * idx - 3 : 3 * idx] = quaternion_to_axis_angle(thumb_input)
            # elif joint.startswith("Thumb_MCP") or joint.startswith("Thumb_DIP"):
            #     result = quaternion_to_axis_angle(relative_quat)[:, [1, 2, 0]]
            #     # result = result[:, [0, 1, 2]]
            #     # result[:, 1] *= -1
            #     input_args["hand_pose"][:, 3 * idx - 3 : 3 * idx] = result
            # # if joint.startswith("Thumb_CMC"):
            #     result = quaternion_to_axis_angle(relative_quat)[:, [1, 0, 2]]
            #     result = result[:, [0, 1, 2]]
            #     result[:, 1] *= -1
            # if joint.startswith("Thumb_MCP") or joint.startswith("Thumb_DIP"):
            #     result = quaternion_to_axis_angle(relative_quat)[:, [1, 0, 2]]
            #     # result = result[:, [0, 1, 2]]
            #     result[:, 1] *= -1
            # #     # result[:, 0] *= -1
            # #     # result[:, 1] *= -1
            #     input_args["hand_pose"][:, 3 * idx - 3 : 3 * idx] = result
            # else:    
            #     input_args["hand_pose"][:, 3 * idx - 3 : 3 * idx] = quaternion_to_axis_angle(relative_quat)[:, [1, 2, 0]]
            if joint.startswith('Thumb'):
                result = quaternion_to_axis_angle(relative_quat)[:, [2, 0, 1]]
                result[:, 1] *= -1
                input_args["hand_pose"][:, 3 * idx - 3 : 3 * idx] = result
            else:
                input_args["hand_pose"][:, 3 * idx - 3 : 3 * idx] = quaternion_to_axis_angle(relative_quat)[:, [1, 2, 0]]
    input_args["global_orient"] = quaternion_to_axis_angle(store_global_quat["global_orient"])[:, [1, 2, 0]]

    return split_dict_into_batches(input_args, batch) if batch else input_args
