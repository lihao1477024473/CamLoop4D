#!/bin/bash
# =============================================================================
# vgen.sh — 视频生成统一入口（整合 vgen/ 下两个项目）
#
#   1. AC3D   (vgen/ac3d-main)          : 相机可控视频生成 (CogVideoX + ControlNet)
#   2. SEVA   (vgen/stable-virtual-camera-main) : Stable Virtual Camera 单图→轨迹视频
#
# 用法:
#   bash vgen.sh ac3d-2b      # AC3D: CogVideoX-2b (约48GB显存)
#   bash vgen.sh ac3d-5b      # AC3D: CogVideoX-5b (约80GB显存)
#   bash vgen.sh seva         # SEVA: 单图 → 预设轨迹视频
#   bash vgen.sh all          # 顺序执行上述全部
#
# 环境准备(一次性):
#   - AC3D: cd vgen/ac3d-main && pip install -r requirements.txt
#   - SEVA: cd vgen/stable-virtual-camera-main && pip install -e . \
#           && huggingface-cli login   # 需访问 stabilityai/stable-virtual-camera
# =============================================================================

set -e

# ---------------------------------------------------------------------------
# 通用路径配置 (AutoDL 风格, 按需修改)
# ---------------------------------------------------------------------------
# vgen 仓库根目录(脚本所在目录的上两级: lh/ -> 项目根)
VGEN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../vgen" && pwd)"
AC3D_DIR="${VGEN_DIR}/ac3d-main"
SEVA_DIR="${VGEN_DIR}/stable-virtual-camera-main"

# 数据/输出根目录
DATA_ROOT="/root/autodl-tmp/vgen-data"
OUT_ROOT="/root/autodl-tmp/results/vgen"

# AC3D 数据(RealEstate10K 格式: annotations/ pose_files/ video_clips/)
AC3D_DATASET_DIR="${DATA_ROOT}/realestate10k"
AC3D_ANNOTATION="annotations/test.json"

# AC3D ControlNet 权重路径(按实际训练输出修改)
AC3D_CKPT_2B="${AC3D_DIR}/out/2B/checkpoint-10000.pt"
AC3D_CKPT_5B="${AC3D_DIR}/out/5B/checkpoint-10000.pt"

# SEVA 输入单图目录(<data_path>/scene_1.png 等)
SEVA_IMG_DIR="${DATA_ROOT}/seva_images"
# SEVA 预设轨迹: orbit|spiral|lemniscate|zoom-in|zoom-out|dolly zoom-in|
#                dolly zoom-out|move-forward|move-backward|move-up|move-down|
#                move-left|move-right|roll
SEVA_TRAJ="orbit"

# 通用生成提示词(AC3D 用; SEVA 不依赖 prompt)
PROMPT="Three fluffy sheep sit side by side at a rustic wooden table, each eagerly digging into their bowls of spaghetti. The pasta is tangled playfully around their woolly faces, and the bright red sauce splatters across their fur. The scene takes place in a lush, green meadow surrounded by rolling hills, with a few grazing cows in the background."

# ---------------------------------------------------------------------------
# AC3D: 相机可控视频生成
#   bash vgen.sh ac3d-2b|ac3d-5b
# ---------------------------------------------------------------------------
run_ac3d() {
    local model_tag="$1"   # "2b" 或 "5b"
    local model_path
    local ckpt_path
    local out_dir
    local num_attn_heads
    local attn_head_dim

    if [ "$model_tag" = "2b" ]; then
        model_path="THUDM/CogVideoX-2b"
        ckpt_path="${AC3D_CKPT_2B}"
        out_dir="${OUT_ROOT}/ac3d-2b"
        num_attn_heads=4
        attn_head_dim=32
    else
        model_path="THUDM/CogVideoX-5b"
        ckpt_path="${AC3D_CKPT_5B}"
        out_dir="${OUT_ROOT}/ac3d-5b"
        num_attn_heads=4
        attn_head_dim=64
    fi

    echo "========== [AC3D-${model_tag}] 开始生成视频 =========="
    echo "  dataset   : ${AC3D_DATASET_DIR}"
    echo "  base model: ${model_path}"
    echo "  controlnet: ${ckpt_path}"
    echo "  output    : ${out_dir}"

    mkdir -p "${out_dir}"
    cd "${AC3D_DIR}"

    python inference/cli_demo_camera.py \
        --video_root_dir "${AC3D_DATASET_DIR}" \
        --annotation_json "${AC3D_ANNOTATION}" \
        --prompt "${PROMPT}" \
        --base_model_path "${model_path}" \
        --controlnet_model_path "${ckpt_path}" \
        --output_path "${out_dir}" \
        --start_camera_idx 0 \
        --end_camera_idx 7 \
        --stride_min 2 \
        --stride_max 2 \
        --controlnet_weights 1.0 \
        --controlnet_guidance_start 0.0 \
        --controlnet_guidance_end 0.4 \
        --controlnet_transformer_num_attn_heads "${num_attn_heads}" \
        --controlnet_transformer_attention_head_dim "${attn_head_dim}" \
        --controlnet_transformer_out_proj_dim_factor 64 \
        --controlnet_transformer_out_proj_dim_zero_init

    echo "========== [AC3D-${model_tag}] 完成, 输出: ${out_dir}/*_out.mp4 =========="
}

# ---------------------------------------------------------------------------
# SEVA: Stable Virtual Camera 单图 → 预设轨迹视频
#   bash vgen.sh seva
# ---------------------------------------------------------------------------
run_seva() {
    echo "========== [SEVA] 单图 → 轨迹视频 开始 =========="
    echo "  输入图像目录 : ${SEVA_IMG_DIR}"
    echo "  预设轨迹     : ${SEVA_TRAJ}"
    echo "  输出         : ${SEVA_DIR}/work_dirs/demo/"

    cd "${SEVA_DIR}"

    # 参考 docs/CLI_USAGE.md 的 img2trajvid_s-prob 示例
    # - orbit/spiral/lemniscate 适合展示 3D 感
    # - 对 pan/dolly 类平移轨迹, 建议增大 camera_scale (如 10.0)
    python demo.py \
        --data_path "${SEVA_IMG_DIR}" \
        --task img2trajvid_s-prob \
        --replace_or_include_input True \
        --traj_prior "${SEVA_TRAJ}" \
        --cfg 4.0,2.0 \
        --guider 1,2 \
        --num_targets 111 \
        --L_short 576 \
        --use_traj_prior True \
        --chunk_strategy interp

    echo "========== [SEVA] 完成, 输出见 work_dirs/demo/img2trajvid_s-prob/ =========="
}

# ---------------------------------------------------------------------------
# 参数分发
# ---------------------------------------------------------------------------
usage() {
    echo "用法: bash vgen.sh {ac3d-2b|ac3d-5b|seva|all}"
    exit 1
}

case "${1:-}" in
    ac3d-2b)
        run_ac3d "2b"
        ;;
    ac3d-5b)
        run_ac3d "5b"
        ;;
    seva)
        run_seva
        ;;
    all)
        run_ac3d "2b"
        run_ac3d "5b"
        run_seva
        ;;
    *)
        usage
        ;;
esac
