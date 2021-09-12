## SEER Performance Model

SEER is an iteration time prediction model for CNNs, targeting on GPU platforms. For more details of SEER, please refer to our paper (SEER: A Time Prediction Model for CNNs from GPU Kernel’s View).

### Instructions

SEER is a performance model, mainly targeted on cuDNN convolution GPU kernels. We implement the model in MATLAB and collect metrics using `nvprof`. Besides, we also write parser code to parse the raw profiling results into formatted Excel files. The workflow and related command are as following:

1. Compile the data collecting program.

   ```
   cd cuDNN && make
   ```

2. Collect & format data (training set & test set). (using `nvprof`).

   ```
   bash ./SEER_data_collect_cudnn.sh
   ```

   You may need `sudo` to collect some of the metrics. If you find problems, try to add `sudo -E env "PATH=$PATH" "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" ` before the above command.

3. Fit model coefficients on training set. (using MATLAB)

   ```
   (execute this in MATLAB)
   seer_train 
   seer_train_other_ops
   ```

4. Predict time of unseen kernels on test set. (using MATLAB)

   ```
   (execute this in MATLAB)
   seer_evaluate_test_set_I
   seer_evaluate_test_set_II
   ```

Ideally, the code should be runnable and the performance model should work for most NVIDIA GPUs, but we only evaluated it on Titan Xp and Titan V. If you find problems running the data collecting code or parser code, you can also collect the needed data by yourself and organize it as our format (please refer to "Data format"), then use our models in MATLAB to fit the coefficients and evaluate accuracy.

### Data Format

If you would like to collect data by yourself, please format your data as Excel files, and organize all the metrics in this order: (number is column index, start from 1)

1. batch size

2. \# of input channels

3. input width

4. \# of output channels

5. output width

6. convolution kernel (filter) width 

7. stride

8. padding size

9. algorithm index 

   ```
   0: CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM		--- GEMM-I
   1: CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM	--- GEMM-P
   2: CUDNN_CONVOLUTION_FWD_ALGO_GEMM			--- GEMM
   3: CUDNN_CONVOLUTION_DIRECT		--- this one is ignored in our model.
   4: CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD			--- WINO
   5: CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED		--- WINO-N
   6: CUDNN_CONVOLUTION_FWD_ALGO_FFT			--- FFT
   7: CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING		--- FFT-T
   ```

10. execution time

11. `nvprof metric` inst_fp_32

12. `nvprof metric` dram_read_transactions

13. `nvprof metric` dram_write_transactions

14. \# of maximum number of thread blocks.

    Obtained from NVIDIA CUDA Occupancy Calculator, please refer to "CUDA Occupancy Calculator".

15. \# of launched thread blocks.

16. \# of GPU iterations (waves).

17. `nvprof metric` single_precision_fu_utilization

18. `nvprof metric` dram_utilization

### Things to be changed when applied in a new Hardware/Software

- Kernel names

  cuDNN libraries may have different kernel implementation on different hardware and software versions. Our parser code will parse the metrics of target kernels through kernel names. So please make sure you know the kernel names and modify it correctly in the parser code.

  How to find the names: Run some microbenchmark and profile it using `nvprof --metrics`. Find the kernel names in the profiling results (please use the `--metrics` option to get the truncated kernel names). One convolution algorithm may have multiple implementations, try to run more cases to collect all the possible kernel names.

  Where to modify in the parser code:  in `parse_src_data.py` L18 ~ L22. Replace the kernel names with the names in your hardware/software.

- \# of maximum number of thread blocks.

  \# of maximum number of thread blocks is obtained from CUDA Occupancy Calculator, please manually get these values (please refer to "CUDA Occupancy Calculator") and replace the values in   `parse_src_data.py` L34 ~ L49 with values of your target kernels.

###CUDA Occupancy Calculator

As we described in the paper, for most kernels in our experiments, B_max (Max # of blocks executing in 1 full wave) equals to the number of resident blocks in one GPU. We have provided these number in the data collect script for Titan Xp. If you want to calculate the number by yourself, please follow the steps:

Download CUDA Occupancy Calculator at https://docs.nvidia.com/cuda/cuda-occupancy-calculator/index.html. 

For any interested kernel: 

- run `nvprof --print-gpu-trace ./target_kernel`
- We can get threads per block, registers per thread and user shared memory per block in the nvprof result. 
- Open `CUDA_Occupancy_Calculator.xls` , select GPU capability in part (1), fill in the above three metrics in part (2), then you'll get the max number of blocks per SM in B48-B50 (the smallest) one, then multiply this number with number of SMs (30 for Titan Xp), you'll get the B_max used in our model. 
- For each variant of each algorithm, this only needs to be done once. 

### Some details to denote

- The data collecting process may take a long time (~ 1 day on Titan Xp), because we use nvprof, which will replay the kernels multiple times to get all the metrics. 
- Dynamic Metrics may have a large variance (as described in our paper). If you find the accuracy is not very good, you may unmark the `warmup()` function in `cuDNN/collect_with_algo.cu` and re-compile the program. The warmup function add extra memcpy to flush the cache, it helps to make the dram access times more stable. 

