src_data_set_path=data/src/
tmp_results_path=data/tmp_results/
profile_result_path=data/profiled/
log_file=data_collect_log
exe_file=./cuDNN/collect_with_algo

repeat_times=5

# Generate runnable kernel configurations
for algo in "algo0" "algo1" "algo2" "algo4" "algo5" "algo6" "algo7" "algo5-ctree"
do
python3 ./cuDNN/generate_ops.py $algo $src_data_set_path $exe_file 2>&1 | tee -a $log_file
done

# Collect Benchmark dataset & Test-set-I
for algo in "algo0" "algo1" "algo2" "algo6" "algo7"
do
for i in $(seq 0 ${repeat_times}) 
do
# collect execution time, repeated several times to calculate average time.
nvprof --print-gpu-trace --log-file ${tmp_results_path}nvprof_trace_${algo}_${i} ./cuDNN/collect_with_algo ${src_data_set_path}prof-ops-${algo}.txt
done
# collect metrics, only profile once.
nvprof --print-gpu-trace --metrics inst_fp_32,dram_read_transactions,dram_write_transactions,single_precision_fu_utilization,dram_utilization --log-file ${tmp_results_path}nvprof_metrics_${algo} ./cuDNN/collect_with_algo ${src_data_set_path}prof-ops-${algo}.txt
done

for algo in "algo4" "algo5"
do
for i in $(seq 0 ${repeat_times}) 
do
./cuDNN/collect_with_algo ${src_data_set_path}prof-ops-${algo}.txt > ${tmp_results_path}time_results_${algo}_${i}
done
done

# ctree for Algo5
algo="algo5-ctree"
nvprof --print-gpu-trace --metrics single_precision_fu_utilization,dram_utilization --log-file ${tmp_results_path}nvprof_metrics_${algo} ./cuDNN/collect_with_algo ${src_data_set_path}prof-ops-${algo}.txt

# Test-set-II
# ./cuDNN/collect_without_algo ${src_data_set_path}Test-set-II.txt > ${tmp_results_path}Test-set-II

# Algo-pick-set
echo "======> collecting Algo-pick-set ......"
./cuDNN/collect_without_algo > ${tmp_results_path}Algo-pick-set

# data parsing
for algo in "algo0" "algo1" "algo2" "algo4" "algo5" "algo6" "algo7" "algo2-pre-kernel" "algo7-pre-kernel" "algo5-ctree" "Algo-pick-set"
do
python3 parse_src_data.py ${algo} 2>&1 | tee -a $log_file
done