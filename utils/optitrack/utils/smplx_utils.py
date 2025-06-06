import sys
import os
import torch

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import smplx

# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


def get_smplx_model(model_type="smpl", gender="neutral"):
    model_folder = "Optitrack2SMPL/SMPL"

    if model_type not in ("flame", "mano", "smpl", "smplh", "smplx"):
        raise ValueError(
            f"model_type '{model_type}' is not supported. It should exist in \('flame', 'mano', 'smpl', 'smplh', 'smplx'\)"
        )

    extra_params = {}
    if model_type in ("smplx", "smplh", "mano", "flame"):
        extra_params["use_pca"] = False
        extra_params["use_face_contour"] = True
        extra_params["flat_hand_mean"] = True

    SMPL_class = getattr(smplx, model_type.upper())

    if model_type == "mano":
        model = SMPL_class(
            os.path.join(
                model_folder, model_type, f"MANO_{gender.upper()}.pkl"
            ),
            **extra_params,
        ).to(device)
    else:
        model = SMPL_class(
            os.path.join(model_folder, model_type),
            gender=gender,
            **extra_params,
        ).to(device)

    return (model, model_type)


def get_model_out(model, input_args=None):
    model, model_type = model

    if input_args == None:
        betas_num = 16 if model_type in ("smplx", "smplh") else 10
        betas = torch.zeros(1, betas_num, device=device)
        global_orient = torch.zeros(1, 3, device=device)
        input_args = {}
        if model_type == "smpl":
            input_args = {
                "body_pose": torch.zeros(
                    1, model.NUM_BODY_JOINTS * 3, device=device
                )
            }
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
        elif model_type == "mano":
            input_args = {
                "hand_pose": torch.zeros(
                    1, model.NUM_HAND_JOINTS * 3, device=device
                )
            }
        elif model_type == "flame":
            input_args = {
                "expression": torch.zeros(1, 10, device=device),
                "jaw_pose": torch.zeros(1, 3, device=device),
                "neck_pose": torch.zeros(1, 3, device=device),
                "leye_pose": torch.zeros(1, 3, device=device),
                "reye_pose": torch.zeros(1, 3, device=device),
            }
        model_output = model(
            global_orient=global_orient, betas=betas, **input_args
        )
    else:
        betas_num = 16 if model_type in ("smplx", "smplh") else 10
        betas = torch.zeros(
            input_args["global_orient"].size(0), betas_num, device=device
        )
        model_output = model(betas=betas, **input_args)

    print(
        f"You are using {model_type} - NUM_BODY_JOINTS {model.NUM_BODY_JOINTS}, NUM_JOINTS {model.NUM_JOINTS}, NUM_BETAS {model.num_betas}"
    )

    #  for k, v in model_output.items():
    #      if isinstance(v, torch.Tensor):
    #          print(f"{model_type}-{k}: {v.shape}")

    return model_output
