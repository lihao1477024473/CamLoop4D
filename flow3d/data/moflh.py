# 自定义脚本：改动函数

'''
注释: lihao-mof

修改的地方
casual_dataset.py
1. 函数get_tracks_3d, line=341

'''

import os
import sys

# 获取当前文件的上两级目录（即 project_root）
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # 上一级目录（project_root）
sys.path.append(parent_dir)



import numpy as np
import torch
import torch.nn.functional as functional
# from flow3d.transforms import rt_to_mat4,rmat_to_cont_6d,cont_6d_to_rmat
from transforms import rt_to_mat4,rmat_to_cont_6d,cont_6d_to_rmat


# NUM_FRAME = 277 # 48/60; backpack=180,paper-windmill=696,277
# num_base = 10 # 25;10

w2c_lastFrame =  torch.tensor(
    [[1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]],dtype=torch.float32)


def compare_w2c(w2c,w2c_lastFrame):
    w2c = w2c.detach().cpu()
    compare_isclose = np.isclose(w2c,w2c_lastFrame)
    compare_allclose = np.allclose(w2c,w2c_lastFrame, atol=1e-7)
    # print(f"w2c: \n{w2c} \nw2c_lastFrame: \n{w2c_lastFrame}")
    # print(f"w2c: \n{w2c}")
    # print(f"nw2c_lastFrame: \n{w2c_lastFrame}")
    # print(f"compare_allclose: {compare_allclose} \ncompare_isclose: \n{compare_isclose}")
    print(f"-----------------------------------------------------------compare_allclose: {compare_allclose}")


def uniform_sample(data, sample_rate):
    """
    对输入列表 data 按照指定的采样率进行均匀采样。
    参数:
        data (list): 输入列表，长度为 N
        sample_rate (float): 采样率，>0
    返回:
        list: 均匀采样后的新列表
    """
    if not data or sample_rate <= 0:
        raise ValueError("data 不能为空且 sample_rate 必须大于 0")
    N = len(data)
    # 新序列的长度由采样率决定
    new_length = int(np.round(N * sample_rate))
    # 使用 numpy.linspace 创建索引映射
    indices = np.linspace(0, N - 1, new_length)
    # 使用 numpy.floor + 插值（可选）
    # 这里使用最近邻插值，也可以换成线性插值等
    sampled_data = [data[int(np.floor(i))] for i in indices]
    return sampled_data

# 张量采样：不能超过指定的点数
def uniform_sample_points(tensor: torch.Tensor, threshold: int = 50000) -> torch.Tensor:
    N = tensor.shape[0]
    device=tensor.device
    indices = torch.arange(N,device=device)
    if N <= threshold:
        return indices
    else:
        rate = threshold/N
        N_target = int(round(N * rate))
        N_target = max(1, N_target)  # 至少保留 1 个点

        indices = torch.linspace(0, N - 1, N_target, dtype=torch.long, device=tensor.device) # 生成均匀索引（在 GPU 上）
        indices = torch.unique(indices)  # 防止浮点误差导致重复
        return indices

# 张量采样：动态高斯不能超过指定的点数;fg + bg
def uniform_sample_points_v2(tensor: torch.Tensor, threshold_fg: int = 100000,type_point="fg",enable: bool =False) -> torch.Tensor:
    N = tensor.shape[0]
    device=tensor.device
    indices = torch.arange(N,device=device)
    # 不启用采样
    if enable == False:
         return indices
    
    # 启用采样
    # b = type_point == "fg"
    # print(f"N={N} | threshold_fg={threshold_fg} {b} {type_point}")
    if type_point == "fg":
        # print(f"N={N} | threshold_fg={threshold_fg}")
        if N <= threshold_fg:
            return indices
        else:
            # print("--------")
            rate = threshold_fg/N
            N_target = int(round(N * rate))
            N_target = max(1, N_target)
            indices = torch.linspace(0, N - 1, N_target, dtype=torch.long, device=tensor.device) # 生成均匀索引（在 GPU 上）
            indices = torch.unique(indices)
            return indices
    else:
        return indices

# 张量采样：动态高斯不能超过指定的点数;fg + bg
def random_sample_indices(tensor: torch.Tensor, threshold_fg: int = 100000,type_point="fg",enable: bool =False) -> torch.Tensor:
    N = tensor.shape[0]
    device=tensor.device
    indices = torch.arange(N,device=device)
    # 不启用采样
    if enable == False:
         return indices
    
    # 启用采样
    # b = type_point == "fg"
    # print(f"N={N} | threshold_fg={threshold_fg} {b} {type_point}")
    if type_point == "fg":
        # print(f"N={N} | threshold_fg={threshold_fg}")
        if N <= threshold_fg:
            return indices
        else:
            # print("--------")
            rate = threshold_fg/N
            N_target = int(round(N * rate))
            N_target = max(1, N_target)
            # 均匀
            # indices = torch.linspace(0, N - 1, N_target, dtype=torch.long, device=tensor.device) # 生成均匀索引（在 GPU 上）
            #随机打乱所有索引，取前 N_target 个
            indices = torch.randperm(N, device=tensor.device)[:N_target]
            indices = torch.unique(indices)
            return indices
    else:
        return indices


# 张量随机采样：超过指定点按比率随机采样
def random_sample_points(tensor: torch.Tensor, rate: float, threshold: int = 50000) -> torch.Tensor:
    """
    对 [N, F, B] 的 tensor 沿点维度 N 进行随机均匀采样。
    仅当 N > threshold 时才采样，否则直接返回原张量。

    Args:
        tensor (torch.Tensor): shape [N, F, B], on CUDA
        rate (float): 采样率，0 < rate <= 1
        threshold (int): 触发采样的点数阈值，默认 50000

    Returns:
        torch.Tensor: shape [N_out, F, B] if N > threshold, else [N, F, B]
    """
    assert tensor.dim() == 3, "Input tensor must be of shape [N, F, B]"
    assert 0 < rate <= 1, "Sampling rate must be in (0, 1]"

    N, F, B = tensor.shape

    # ✅ 判断点数量是否超过阈值
    if N <= threshold:
        return tensor  # 不采样，原样返回

    # 开始随机采样
    N_target = int(round(N * rate))
    N_target = max(1, N_target)  # 至少保留 1 个点

    if N_target == N:
        return tensor

    # 在 GPU 上生成随机排列并取前 N_target 个索引
    indices = torch.randperm(N, device=tensor.device)[:N_target]

    # 可选：保持索引顺序（如原始顺序），便于后续处理
    indices = indices.sort().values

    return tensor[indices]  # [N_target, F, B]

def init_motion_coefs(N,F,B,alg="randn"):
    # # 250615
    # coefs = torch.randn(N, F, B) # 初始化为标准正态分布 N(0, 1)
    # coefs = torch.sigmoid(coefs) # 使用 sigmoid 将其映射到 (0, 1) 区间
    # # coefs =  functional.normalize(coefs,dim=-1) # 归一化
    # # return 10*coefs
    # return 1.0*coefs # 250617,放大效果不佳，改回原来的系数

    # 250625
    coefs = torch.randn(B)
    coefs = torch.sigmoid(coefs)
    coefs =coefs.reshape(1,1,B).repeat(N, F, 1)
    return 1.0*coefs


def compute_coef(coef1,base1,base2):
    '''
    coef1.shape = [N, B]
    base1.shape = [B, F, 6]
    coef2.shape = [B, 6]
    base2.shape = [N, F, B]

    torch.einsum("pk,kni->pni", coef1, base1)=torch.einsum("pnk,ki->pni", coef2, base2)
    '''
    # 第一步：计算左边的 einsum
    left = torch.einsum("pk,kni->pni", coef1, base1)
    # 第二步：计算 base2 的伪逆
    base2_pinv = torch.linalg.pinv(base2)
    # 第三步：计算 c2
    coef2 = torch.einsum("pni,ij->pnj", left, base2_pinv)

    # 检查 c2 形状是否符合预期
    print("coef2.shape:", coef2.shape)  # 应输出: torch.Size([10, 5, 3])

    # 验证等式是否成立（可选）
    right = torch.einsum("pnk,ki->pni", coef2, base2)  # shape: [10, 5, 6]

    # 检查误差
    error = torch.norm(left - right).item()
    print("[left - right]误差（应很小）:", error)

    return coef2


def check_bases_sizes(rots, transls):
    # 示例函数，确保 rots 和 transls 尺寸匹配
    return rots.shape[0] == transls.shape[0]

# 六个基本运动基的 4x4 矩阵表示（在 se(3) 中）
# 三个平移基
# 沿x轴的平移
t_x = np.array(
    [[0,0,0,1],
     [0,0,0,0],
     [0,0,0,0],
     [0,0,0,0]]
)
# 沿y轴的平移
t_y = np.array(
    [[0,0,0,0],
     [0,0,0,1],
     [0,0,0,0],
     [0,0,0,0]]
)

# 沿z轴的平移
t_z = np.array(
    [[0,0,0,0],
     [0,0,0,0],
     [0,0,0,1],
     [0,0,0,0]]
)

# 三个旋转基
# 绕x轴的旋转
r_x = np.array(
    [[1,0,0,0],
     [0,0,-1,0],
     [0,1,0,0],
     [0,0,0,0]]
)
# 绕y轴的旋转
r_y = np.array(
    [[0,0,1,0],
     [0,1,0,0],
     [-1,0,0,0],
     [0,0,0,0]]
)
# 绕z轴的旋转
r_z = np.array(
    [[0,-1,0,0],
     [1,0,0,0],
     [0,0,1,0],
     [0,0,0,0]]
)

# 单位阵
e_mat = np.array(
    [[1,0,0,0],
     [0,1,0,0],
     [0,0,1,0],
     [0,0,0,0]]
)
device = "cuda"

# # 方式1：3个基（三平移，三旋转）
# num_base = 3
# rots_mat = torch.from_numpy(np.array([r_x,r_y,r_z],dtype=np.float32))
# rots_6d = rmat_to_cont_6d(rots_mat[:,:3,:3]).to(device)
# transls_3d = torch.from_numpy(np.array([t_x,t_y,t_z],dtype=np.float32))[:,:3,-1].to(device)
# print(f"rots_mat:\n{rots_mat}")
# print(f"rots_6d:\n{rots_6d} {rots_6d.shape}")
# print(f"transls_3d:\n{transls_3d} {transls_3d.shape}")



# 方式2：6个基 + x
# bases_l = [t_x,t_y,t_z,r_x,r_y,r_z]
# for i in range(num_base - len(bases_l)):
#     bases_l.append(e_mat)
# bases = torch.from_numpy(np.array(bases_l,dtype=np.float32))
# rots_6d = rmat_to_cont_6d(bases[:,:3,:3]).to(device)
# transls_3d = bases[:,:3,-1].to(device)
# print(f"\nbases:\n{bases} {bases.shape}")
# print(f"rots_6d:\n{rots_6d} {rots_6d.shape}")
# print(f"transls_3d:\n{transls_3d} {transls_3d.shape}")

# 方式2：6个基（t_x,t_y,t_z,r_x,r_y,r_z） + x（单位阵+0向量）
def get_rots6d_transls3d(num_bases):
    bases_l = [t_x,t_y,t_z,r_x,r_y,r_z] # 6个基（三平移+三旋转）
    for i in range(num_bases - len(bases_l)):
        bases_l.append(e_mat) # 添加
    
    bases = torch.from_numpy(np.array(bases_l,dtype=np.float32))
    rots_6d = rmat_to_cont_6d(bases[:,:3,:3]).to(device)
    transls_3d = bases[:,:3,-1].to(device)
    # print(f"\nbases:\n{bases} {bases.shape}")
    # print(f"rots_6d:\n{rots_6d} {rots_6d.shape}")
    # print(f"transls_3d:\n{transls_3d} {transls_3d.shape}")
    return rots_6d,transls_3d

# 方式3：q组固定基+r个可变基
# quotient, remainder = divmod(num_bases, b)：商和余数
def get_rots6d_transls3d_v2(num_bases):
    bases_fixed = [t_x,t_y,t_z,r_x,r_y,r_z] # 6个基（三平移+三旋转）
    
    quotient, remainder = divmod(num_bases, len(bases_fixed))
    print(f"num_bases: {num_bases} | quotient: {quotient} | remainder: {remainder}")      # 输出: 商: 3
    
    # 固定基:quotient组，quotient*6
    bases =  bases_fixed*quotient
    # print(f"bases: {bases}")
    # print(bases_fixed[0]==bases_fixed[6])
    
    # 可变基：remainder
    for i in range(remainder):
        bases.append(e_mat) # 添加
        
    # print(f"bases: {bases} | bases_len:: {len(bases)}")
    # print(bases[0]==bases[6])
    
    bases = torch.from_numpy(np.array(bases,dtype=np.float32))
    rots_6d = rmat_to_cont_6d(bases[:,:3,:3]).to(device)
    transls_3d = bases[:,:3,-1].to(device)
    # print(f"\nbases:\n{bases} {bases.shape}")
    # print(f"rots_6d:\n{rots_6d} {rots_6d.shape}")
    # print(f"transls_3d:\n{transls_3d} {transls_3d.shape}")
    return rots_6d,transls_3d

    
    
    
    
# get_rots6d_transls3d_v2(14) 
    

# coefs = init_motion_coefs(18936,80,num_base).to(device)
# print(f"coefs:\n{coefs} {coefs.shape}")
# print(f"coefs:",coefs.min(), coefs.max())  # 输出大致在 (0, 1) 范围内

# num_frames = 0
# ts = torch.arange(0, num_frames, device=device)
# transls = transls_3d[:, ts]  # (K, 3)
# print("transls:",transls,transls.shape)
# exit(-1)

# 测试
# transls = torch.einsum("pnk,ki->pni", coefs, transls_3d) # (G, B, 3)
# transls = torch.einsum("pnk,ki->pni", coefs, rots_6d) # (G, B, 3
# rots = torch.einsum("pnk,ki->pni", coefs, rots_6d)  # (G, B, 6)
# print(f"[moflh] transls={transls.shape} | rots={rots.shape}")




