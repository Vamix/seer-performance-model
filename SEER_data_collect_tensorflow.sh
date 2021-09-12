src_data_set_path=data/src/
tmp_results_path=data/tmp_results/
profile_result_path=data/profiled/
log_file=data_collect_log

DATASET_DIR=/tmp/flowers
TRAIN_DIR=/tmp/flowers-models/
repeat_times=4

# Maxpool op (TensorFlow)
echo "======> collecting Maxpool in TensorFlow ......"
python3 ./TensorFlow/other_ops/main.py maxpool

# Conv grad (TensorFlow)
echo "======> collecting Conv (backward) in TensorFlow ......"
python3 ./TensorFlow/other_ops/main.py conv_grad

# 1x1 Conv op (TensorFlow)
echo "======> collecting 1x1 Conv in TensorFlow ......"
python3 ./TensorFlow/other_ops/main.py conv_1x1

# Layout transformation (TensorFlow)
echo "======> collecting layout_transform op in TensorFlow ......"
nvprof --print-gpu-trace --log-file ${tmp_results_path}layout python3 ./TensorFlow/other_ops/main.py layout > ${tmp_results_path}layout_ops
python3 parse_src_data.py layout-trans 2>&1 | tee -a $log_file
