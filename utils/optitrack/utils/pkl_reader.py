import numpy as np
import pickle
from scipy.spatial.transform import Rotation as R

def calculate_thumb_axes(joints, kintree_table, thumb_joint_idx):
    """
    计算大拇指初始局部坐标系的各轴方向
    :param joints: 各关节的位置，形状为 (n_joints, 3)
    :param kintree_table: 骨架层级，定义每个关节的父子关系
    :param thumb_joint_idx: 大拇指关节的索引
    :return: 大拇指局部坐标系的三个单位向量(x, y, z)
    """
    parent_idx = kintree_table[0, thumb_joint_idx]
    
    if parent_idx == -1:
        raise ValueError("大拇指关节没有父关节！")
    
    # 计算 x 轴：从父关节指向当前关节
    x_axis = joints[thumb_joint_idx] - joints[parent_idx]
    x_axis /= np.linalg.norm(x_axis)
    
    # 假设掌心方向（全局坐标系的 z 轴方向）为 [0, 0, 1]
    palm_z = np.array([0, 0, 1])
    
    # 计算 z 轴：与掌心方向垂直
    z_axis = np.cross(x_axis, palm_z)
    z_axis /= np.linalg.norm(z_axis)
    
    # 计算 y 轴：通过右手法则确定
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    
    return x_axis, y_axis, z_axis

def calculate_global_rotation(thumb_axes):
    """
    计算大拇指局部坐标系相对于全局坐标系的旋转
    :param thumb_axes: 大拇指局部坐标系的三个单位向量(x, y, z)
    :return: 大拇指局部坐标系相对于全局坐标系的旋转（四元数和轴角表示）
    """
    # 提取局部坐标系的三个轴
    x_axis, y_axis, z_axis = thumb_axes
    
    # 构造大拇指局部旋转矩阵
    R_thumb = np.column_stack((x_axis, y_axis, z_axis))  # 将 x, y, z 轴拼成旋转矩阵
    
    # 检查是否为正交矩阵
    if not np.allclose(np.dot(R_thumb.T, R_thumb), np.eye(3), atol=1e-6):
        raise ValueError("构造的旋转矩阵不是正交矩阵，请检查输入的轴定义。")
    
    # 将旋转矩阵转换为四元数
    quat = R.from_matrix(R_thumb).as_quat()  # 四元数 [x, y, z, w]
    
    # 将旋转矩阵转换为轴角
    axis_angle = R.from_matrix(R_thumb).as_rotvec()  # 轴角 [rx, ry, rz]
    
    return {
        "rotation_matrix": R_thumb,
        "quaternion": quat,
        "axis_angle": axis_angle
    }


with open('./SMPL/mano/MANO_RIGHT.pkl', 'rb') as f:
    mano_data = pickle.load(f, encoding="latin1")

# 提取关节位置信息和骨架层级
joints = mano_data['J']  # (n_joints, 3)
kintree_table = mano_data['kintree_table']

thumb_joint_idx = 14

# 计算大拇指初始坐标系
x_axis, y_axis, z_axis = calculate_thumb_axes(joints, kintree_table, thumb_joint_idx)

rotation_result = calculate_global_rotation((x_axis, y_axis, z_axis))
thumb_rotation = rotation_result["quaternion"]
np.save("thumb_CMC_coordinate.npy", thumb_rotation)

print("大拇指初始局部坐标系:")
print("x 轴方向:", x_axis)
print("y 轴方向:", y_axis)
print("z 轴方向:", z_axis)

print("大拇指局部坐标系相对于全局坐标系的旋转矩阵:")
print(rotation_result["rotation_matrix"])

print("\n大拇指局部坐标系相对于全局坐标系的四元数表示:")
print(rotation_result["quaternion"])

print("\n大拇指局部坐标系相对于全局坐标系的轴角表示:")
print(rotation_result["axis_angle"])

print(mano_data['J'])
