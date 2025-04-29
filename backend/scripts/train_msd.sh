#!/bin/bash

# MSD数据集训练脚本
# 使用: ./train_msd.sh [h5文件目录] [triplets文件路径] [输出模型路径]

# 确保脚本是可执行的:
# chmod +x train_msd.sh

# 设置日志文件
LOG_FILE="./msd_training_$(date +%Y%m%d_%H%M%S).log"
echo "开始日志记录到 $LOG_FILE" > $LOG_FILE

# 显示用法
if [ "$#" -lt 2 ]; then
    echo "用法: $0 <h5文件目录> <triplets文件路径> [输出模型路径]" | tee -a $LOG_FILE
    echo "例如: $0 /path/to/msd/h5 /path/to/triplets.txt ../models/msd_model.pkl" | tee -a $LOG_FILE
    exit 1
fi

# 设置参数
H5_DIR="$1"
TRIPLETS_FILE="$2"
OUTPUT_PATH="${3:-../models/msd_model.pkl}"  # 默认输出路径

echo "==== MSD 数据集训练 ====" | tee -a $LOG_FILE
echo "H5文件目录: $H5_DIR" | tee -a $LOG_FILE
echo "Triplets文件: $TRIPLETS_FILE" | tee -a $LOG_FILE
echo "输出模型路径: $OUTPUT_PATH" | tee -a $LOG_FILE
echo "=============================" | tee -a $LOG_FILE

# 检查目录和文件是否存在
if [ ! -d "$H5_DIR" ]; then
    echo "错误: H5文件目录不存在: $H5_DIR" | tee -a $LOG_FILE
    exit 1
fi

if [ ! -f "$TRIPLETS_FILE" ]; then
    echo "错误: Triplets文件不存在: $TRIPLETS_FILE" | tee -a $LOG_FILE
    exit 1
fi

# 确保输出目录存在
OUTPUT_DIR=$(dirname "$OUTPUT_PATH")
mkdir -p "$OUTPUT_DIR"

# 获取脚本目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TRAIN_SCRIPT="$SCRIPT_DIR/../models/create_msd_model.py"

# 检查训练脚本是否存在
if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "错误: 训练脚本不存在: $TRAIN_SCRIPT" | tee -a $LOG_FILE
    exit 1
fi

# 执行Python脚本
echo "执行Python脚本: $TRAIN_SCRIPT" | tee -a $LOG_FILE
python "$TRAIN_SCRIPT" \
    --h5_dir "$H5_DIR" \
    --triplets_file "$TRIPLETS_FILE" \
    --output "$OUTPUT_PATH" \
    --max_songs_per_user 20 \
    --test_size 0.2 2>&1 | tee -a $LOG_FILE

# 检查执行结果
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "训练成功完成！模型已保存到 $OUTPUT_PATH" | tee -a $LOG_FILE
else
    echo "训练过程中出错，请检查日志" | tee -a $LOG_FILE
    exit 1
fi 