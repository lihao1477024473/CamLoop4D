#!/bin/bash
# =============================================================================
# sharedCamTrajectory.sh — 共享相机轨迹接口（生成 ↔ 重建）
#
# 封装 sharedCamTrajectory.py 的常用指令:
#   plucker-trajectory      生成指定类型的相机轨迹 + Plücker 嵌入, 保存共享相机文件
#   convert                 把已有轨迹(c2w .npy)转为重建端相机文件
#   realestate10k-to-iphone 把 RealEstate10K pose 文件转为 iPhone 相机 JSON
#   inspect                 查看共享相机文件
#
# 用法:
#   bash sharedCamTrajectory.sh plucker-trajectory [extra args]
#   bash sharedCamTrajectory.sh convert [extra args]
#   bash sharedCamTrajectory.sh realestate10k-to-iphone [extra args]
#   bash sharedCamTrajectory.sh inspect --cam-file <file>
#   bash sharedCamTrajectory.sh all          # 依次执行常用示例
#
# 依赖:
#   - Python 环境需安装 numpy (如 miniconda: C:/Users/Administrator/miniconda3/python.exe)
# =============================================================================

set -e

# ---------------------------------------------------------------------------
# 路径配置 (AutoDL 风格, 按需修改)
# ---------------------------------------------------------------------------
# 项目根目录(脚本所在目录的上一级: lh/ -> 项目根)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="${ROOT_DIR}/sharedCamTrajectory.py"

# Python 解释器(需含 numpy; 本地测试用 miniconda, AutoDL 可改为 conda 环境)
PYTHON="${PYTHON:-python}"

# 数据根目录
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/data}"

# RealEstate10K 数据(annotations/ pose_files/ video_clips/)
RE10K_DIR="${RE10K_DIR:-${DATA_ROOT}/realestate10k}"
RE10K_POSE_FILE="${RE10K_POSE_FILE:-${RE10K_DIR}/pose_files/0000cc6d8b108390.txt}"

# 默认序列名 / 输出目录
SEQ_NAME="${SEQ_NAME:-my_scene}"
OUT_DIR="${OUT_DIR:-${DATA_ROOT}/seva2som}"

# 默认相机内参
FX="${FX:-700.0}"
FY="${FY:-700.0}"
CX="${CX:-360.0}"
CY="${CY:-240.0}"
H="${H:-480}"
W="${W:-720}"

# ---------------------------------------------------------------------------
# 1) 生成相机轨迹 (Plücker)
# ---------------------------------------------------------------------------
run_plucker_trajectory() {
    echo "========== [sharedCamTrajectory] plucker-trajectory =========="
    echo "  seq-name: ${SEQ_NAME}   out-dir: ${OUT_DIR}"
    echo "  traj    : spiral, ${H}x${W}, fx/fy=${FX}/${FY}, cx/cy=${CX}/${CY}"

    "${PYTHON}" "${SCRIPT}" plucker-trajectory \
        --traj-type spiral \
        --num-frames 80 \
        --fx "${FX}" --fy "${FY}" --cx "${CX}" --cy "${CY}" \
        --height "${H}" --width "${W}" \
        --out-dir "${OUT_DIR}" \
        --seq-name "${SEQ_NAME}"
}

# ---------------------------------------------------------------------------
# 2) 转换已有轨迹 -> 重建端相机文件
# ---------------------------------------------------------------------------
run_convert() {
    TRAJ_C2W="${TRAJ_C2W:?需指定 --traj-c2w <c2w.npy>}"
    echo "========== [sharedCamTrajectory] convert =========="
    echo "  traj-c2w: ${TRAJ_C2W}"
    echo "  intrinsics: ${FX},${FY},${CX},${CY}   img-shape: ${H},${W}"

    "${PYTHON}" "${SCRIPT}" convert \
        --traj-c2w "${TRAJ_C2W}" \
        --intrinsics "${FX},${FY},${CX},${CY}" \
        --img-shape "${H},${W}" \
        --out-dir "${OUT_DIR}" \
        --seq-name "${SEQ_NAME}"
}

# ---------------------------------------------------------------------------
# 3) RealEstate10K pose -> iPhone 相机 JSON
# ---------------------------------------------------------------------------
run_re10k_to_iphone() {
    echo "========== [sharedCamTrajectory] realestate10k-to-iphone =========="
    echo "  pose-file: ${RE10K_POSE_FILE}"
    echo "  seq-name : ${SEQ_NAME}   out-dir: ${OUT_DIR}"

    "${PYTHON}" "${SCRIPT}" realestate10k-to-iphone \
        --pose-file "${RE10K_POSE_FILE}" \
        --out-dir "${OUT_DIR}" \
        --seq-name "${SEQ_NAME}"
}

# ---------------------------------------------------------------------------
# 4) 查看共享相机文件
# ---------------------------------------------------------------------------
run_inspect() {
    CAM_FILE="${CAM_FILE:?需指定 --cam-file <file.npy>}"
    echo "========== [sharedCamTrajectory] inspect =========="
    "${PYTHON}" "${SCRIPT}" inspect --cam-file "${CAM_FILE}"
}

# ---------------------------------------------------------------------------
# 参数分发
# ---------------------------------------------------------------------------
usage() {
    echo "用法: bash sharedCamTrajectory.sh {plucker-trajectory|convert|realestate10k-to-iphone|inspect|all}"
    echo "      [额外参数以环境变量传入, 如 SEQ_NAME=xx OUT_DIR=xx bash sharedCamTrajectory.sh ...]"
    exit 1
}

case "${1:-}" in
    plucker-trajectory)
        run_plucker_trajectory
        ;;
    convert)
        run_convert
        ;;
    realestate10k-to-iphone)
        run_re10k_to_iphone
        ;;
    inspect)
        run_inspect
        ;;
    all)
        run_plucker_trajectory
        run_re10k_to_iphone
        ;;
    *)
        usage
        ;;
esac
