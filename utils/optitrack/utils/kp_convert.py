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
    # Check if the variable var_name exists in the global variable dictionary
    if var_name in globals():
        return globals()[var_name]
    else:
        raise NameError(f"Variable '{var_name}' is not defined")


def split_dict_into_batches(data_dict, batch_size):
    # Get fixed N
    # Ensure there's at least one tensor to get the size from
    if not data_dict:
        return []
    N = list(data_dict.values())[0].size(0)
    # Calculate the total number of batches
    num_batches = N // batch_size + (N % batch_size > 0)
    # Store the split dictionaries
    batches = [{} for _ in range(num_batches)]
    # Iterate over the key-value pairs in the dictionary
    for key, value in data_dict.items():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, N)
            # Put the sliced data into the corresponding dictionary
            batches[i][key] = value[start_idx:end_idx]
    return batches


def get_body_pose(df_bone, model_type, human_name, batch=None):
    kpDict = get_variable_value("OPTI2" + model_type.upper())
    num_frames = df_bone.shape[0]

    input_args = {}
    if model_type == "smpl":
        input_args = {
            "global_orient": torch.zeros(num_frames, 3, device=device),
            "body_pose": torch.zeros(num_frames, 23 * 3, device=device)
        }
        # prepare smpl input_args
        for joint, idx in kpDict.items():
            axis_angle = get_joint_axis_angle(df_bone, joint, human_name)
            if idx == 0:
                input_args["global_orient"] = axis_angle
            else:
                input_args["body_pose"][:, 3 * idx - 3 : 3 * idx] = axis_angle

    elif model_type == "smplh":
        # This part remains a TODO as per the original structure
        input_args = {
            "body_pose": torch.zeros(
                num_frames, 51 * 3, device=device # 21 body joints * 3
            ),
            "left_hand_pose": torch.zeros(
                num_frames, 15 * 3, device=device
            ),
            "right_hand_pose": torch.zeros(
                num_frames, 15 * 3, device=device
            ),
        }
        # TODO: prepare smplh input_args
    elif model_type == "smplx":
        # Initialize tensors for SMPL-X based on the number of frames
        input_args = {
            "global_orient": torch.zeros(num_frames, 3, device=device),
            "body_pose": torch.zeros(num_frames, 21 * 3, device=device),
            "left_hand_pose": torch.zeros(num_frames, 15 * 3, device=device),
            "right_hand_pose": torch.zeros(num_frames, 15 * 3, device=device),
            "jaw_pose": torch.zeros(num_frames, 3, device=device),
            "leye_pose": torch.zeros(num_frames, 3, device=device),
            "reye_pose": torch.zeros(num_frames, 3, device=device),
            # FIX: Add expression tensor with correct batch size
            "expression": torch.zeros(num_frames, 10, device=device),
        }
        
        # Populate tensors using the OPTI2SMPLX mapping
        for joint, (tensor_name, idx) in kpDict.items():
            try:
                axis_angle = get_joint_axis_angle(df_bone, joint, human_name)
                
                if tensor_name == "global_orient":
                    input_args["global_orient"] = axis_angle
                else:
                    # For pose tensors, the index is 0-based for joints,
                    # so we map to the correct slice (idx*3 to idx*3+3)
                    # Note: SMPL-X hand pose indices in the model are 1-15, but our mapping is 0-based for the tensor.
                    if 'hand_pose' in tensor_name:
                         # Hand poses are 15*3. The mapping index is 1-based in my previous definition,
                         # let's correct it to be 0-based index for slicing
                         input_args[tensor_name][:, 3 * (idx-1) : 3 * idx] = axis_angle
                    else:
                        input_args[tensor_name][:, 3 * idx : 3 * (idx + 1)] = axis_angle

            except KeyError:
                # This allows processing to continue even if a specific bone is missing from the CSV
                # print(f"Warning: Joint '{joint}' not found in OptiTrack data. Skipping.")
                pass
        
    elif model_type == "mano":
        input_args = {
            "hand_pose": torch.zeros(
                num_frames, 15 * 3, device=device
            )
        }
        # TODO: prepare mano input_args
    elif model_type == "flame":
        input_args = {
            "expression": torch.zeros(num_frames, 10, device=device),
            "jaw_pose": torch.zeros(num_frames, 3, device=device),
            "neck_pose": torch.zeros(num_frames, 3, device=device),
            "leye_pose": torch.zeros(num_frames, 3, device=device),
            "reye_pose": torch.zeros(num_frames, 3, device=device),
        }
        # TODO: prepare flame input_args

    return split_dict_into_batches(input_args, batch) if batch else input_args

def get_parent_joint_idx(idx):
    parent_map = {
        0: None,   # Hand (wrist) has no parent node
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
            
            if joint.startswith('Thumb'):
                result = quaternion_to_axis_angle(relative_quat)[:, [2, 0, 1]]
                result[:, 1] *= -1
                input_args["hand_pose"][:, 3 * idx - 3 : 3 * idx] = result
            else:
                input_args["hand_pose"][:, 3 * idx - 3 : 3 * idx] = quaternion_to_axis_angle(relative_quat)[:, [1, 2, 0]]
    input_args["global_orient"] = quaternion_to_axis_angle(store_global_quat["global_orient"])[:, [1, 2, 0]]

    return split_dict_into_batches(input_args, batch) if batch else input_args