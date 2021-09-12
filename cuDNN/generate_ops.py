import os
import numpy as np
import sys
################################# Profiling configurations #################################
# src_file = "prof_ops-algo4-small.txt"               # source file path
target_algo = sys.argv[1] 
saved_path = sys.argv[2]
exe_file = sys.argv[3]
saved_file = saved_path + "prof-ops-" + target_algo + ".txt"
time_min = 1
time_max = 1000

if target_algo == "algo0":  
    batch_size_range = [128]
    in_chan_range = [3, 16, 64, 128, 512, 1024] 
    in_wid_range = [10, 40, 80, 160, 224, 320]
    out_chan_range = [3, 16, 64, 128, 512, 1024] 
    kernel_range = [1, 3, 5, 7, 9]
    stride_range = [1, 2, 3, 4]
    padding_range = [0]
    algo_range = [0]
elif target_algo == "algo1":  
    batch_size_range = [128]
    in_chan_range = [3, 16, 64, 128, 512, 1024] 
    in_wid_range = [10, 40, 80, 160, 224, 320]
    out_chan_range = [3, 16, 64, 128, 512, 1024] 
    kernel_range = [1, 3, 5, 7, 9]
    stride_range = [1, 2, 3, 4]
    padding_range = [0]
    algo_range = [1]    
elif target_algo == "algo2":  
    batch_size_range = [128]
    in_chan_range = [3, 16, 64, 128, 512, 1024] 
    in_wid_range = [10, 40, 80, 160, 224, 320]
    out_chan_range = [3, 16, 64, 128, 512, 1024] 
    kernel_range = [1, 3, 5, 7, 9]
    stride_range = [1, 2, 3, 4]
    padding_range = [0]
    algo_range = [2]        
elif target_algo == "algo4":
    batch_size_range = [128]
    in_chan_range = [3, 16, 32, 64, 128, 256, 512, 1024] 
    in_wid_range = [16, 32, 48, 64, 96, 128, 224, 256]
    out_chan_range = [3, 16, 32, 64, 128, 256, 512, 1024] 
    kernel_range = [1, 3, 5, 7, 9, 11]
    stride_range = [1]
    padding_range = [0]
    algo_range = [4]
elif target_algo == "algo5":
    batch_size_range = [128]
    in_chan_range = [3, 16, 32, 64, 128, 256, 512, 1024] 
    in_wid_range = [32, 64, 128, 224, 256, 320]
    out_chan_range = [3, 16, 32, 64, 128, 256, 512, 1024] 
    kernel_range = [1, 3, 5, 7, 9, 11]
    stride_range = [1]
    padding_range = [0]
    algo_range = [5]
elif target_algo == "algo5-ctree":
    batch_size_range = [128]
    in_chan_range = [3, 16, 32, 64, 128, 256, 512, 1024] 
    in_wid_range = [32]
    out_chan_range = [3, 16, 32, 64, 128, 256, 512, 1024] 
    kernel_range = [1, 3, 5, 7, 9, 11]
    stride_range = [1]
    padding_range = [0]
    algo_range = [5]
elif target_algo == "algo6":
    batch_size_range = [128]
    in_chan_range = [3, 16, 32, 64, 100, 128, 256, 400, 512, 600, 800, 1024] 
    in_wid_range = [4, 10, 16, 20, 32, 64, 80, 100, 128, 224, 256, 400, 512]
    out_chan_range = [3, 16, 32, 64, 100, 128, 256, 400, 512, 600, 800, 1024] 
    kernel_range = [3]
    stride_range = [1]
    padding_range = [1]
    algo_range = [6]    
elif target_algo == "algo7":
    batch_size_range = [128]
    in_chan_range = [3, 16, 32, 64, 128, 256, 400, 512, 800, 1024] 
    in_wid_range = [4, 10, 16, 32, 64, 100, 128, 224, 256, 400, 512]
    out_chan_range = [3, 16, 32, 64, 128, 256, 400, 512, 800, 1024] 
    kernel_range = [3, 5]
    stride_range = [1]
    padding_range = [1]
    algo_range = [7]        

# if target_algo == "algo0" or target_algo == "algo1" or target_algo == "algo2":  
#     batch_size_range = [128]
#     in_chan_range = [3]#, 16, 64, 128, 512, 1024] 
#     in_wid_range = [10]#, 40, 80, 160, 224, 320]
#     out_chan_range = [3]#, 16, 64, 128, 512, 1024] 
#     kernel_range = [1]#, 3, 5, 7, 9]
#     stride_range = [1]#, 2, 3, 4]
#     padding_range = [0]
#     algo_range = [0]
# elif target_algo == "algo4":
#     batch_size_range = [128]
#     in_chan_range = [3]#, 16, 32, 64, 128, 256, 512, 1024] 
#     in_wid_range = [16]#, 32, 48, 64]
#     out_chan_range = [3]#, 16, 32, 64, 128, 256, 512, 1024] 
#     kernel_range = [1]#, 3, 5, 7, 9, 11]
#     stride_range = [1]
#     padding_range = [0]
#     algo_range = [4]
# elif target_algo == "algo4-large":
#     batch_size_range = [128]
#     in_chan_range = [3]#, 16, 32, 64, 128, 256, 512, 1024] 
#     in_wid_range = [96]#, 128, 224, 256]
#     out_chan_range = [3]#, 16, 32, 64, 128, 256, 512, 1024] 
#     kernel_range = [1]#, 3, 5, 7, 9, 11]
#     stride_range = [1]
#     padding_range = [0]
#     algo_range = [4]
# elif target_algo == "algo5":
#     batch_size_range = [128]
#     in_chan_range = [3]#], 16, 32, 64, 128, 256, 512, 1024] 
#     in_wid_range = [32]
#     out_chan_range = [3]#], 16, 32, 64, 128, 256, 512, 1024] 
#     kernel_range = [1]#, 3, 5, 7, 9, 11]
#     stride_range = [1]
#     padding_range = [0]
#     algo_range = [5]
# elif target_algo == "algo5-ctree":
#     batch_size_range = [128]
#     in_chan_range = [3]#, 16, 32, 64, 128, 256, 512, 1024] 
#     in_wid_range = [64]#, 128, 224, 256, 320]
#     out_chan_range = [3]#, 16, 32, 64, 128, 256, 512, 1024] 
#     kernel_range = [1]#, 3, 5, 7, 9, 11]
#     stride_range = [1]
#     padding_range = [0]
#     algo_range = [5]
# elif target_algo == "algo6":
#     batch_size_range = [128]
#     in_chan_range = [3]#, 16, 32, 64, 100, 128, 256, 400, 512, 600, 800, 1024] 
#     in_wid_range = [4]#, 10, 16, 20, 32, 64, 80, 100, 128, 224, 256, 400, 512]
#     out_chan_range = [3]#, 16, 32, 64, 100, 128, 256, 400, 512, 600, 800, 1024] 
#     kernel_range = [3]
#     stride_range = [1]
#     padding_range = [0]
#     algo_range = [6]    
# elif target_algo == "algo7":
#     batch_size_range = [128]
#     in_chan_range = [3]#, 16, 32, 64, 128, 256, 400, 512, 800, 1024] 
#     in_wid_range = [4]#, 10, 16, 32, 64, 100, 128, 224, 256, 400, 512]
#     out_chan_range = [3]#, 16, 32, 64, 128, 256, 400, 512, 800, 1024] 
#     kernel_range = [3]#, 5]
#     stride_range = [1]
#     padding_range = [0]
#     algo_range = [7]        

################################# generate ops ################################
f_generate_ops = open("tmp/tmp", 'w')
num_ops = len(batch_size_range) * len(in_chan_range) * len(in_wid_range) * len(out_chan_range) * len(kernel_range) * len(stride_range) * len(algo_range)
f_generate_ops.write(str(num_ops) + "\n")
for batch_size in batch_size_range:
    for in_chan in in_chan_range:
        for in_wid in in_wid_range:
            for out_chan in out_chan_range:
                for kernel in kernel_range:
                    for stride in stride_range:
                        for padding in padding_range:
                            for algo in algo_range:
                                f_generate_ops.write(str(batch_size)+"\t"+str(in_chan)+"\t"+str(in_wid)+"\t"+str(out_chan)+"\t0\t"+str(kernel)+"\t"+str(stride)+"\t"+str(padding)+"\t"+str(algo)+"\n")
f_generate_ops.close()
print("====> generating op configs from user input... save to "+ saved_file)
os.system(exe_file + " tmp/tmp > tmp/tmp_log_file_time")
# os.system("nvprof --print-gpu-trace " + " --log-file tmp/tmp_nvprof " + exe_file + " tmp/tmp | tee tmp/tmp_log_file_time")
f_ops = open("tmp/tmp_2", 'w')
effective_ops = []
num_ops = 0
with open("tmp/tmp_log_file_time") as logs:
    for line in logs:
        splits = line.split("\t")
        current_op_time = float(splits[9])
        if(current_op_time > time_min and current_op_time < time_max):
            num_ops += 1
            effective_ops.append("\t".join(splits[:-1])+"\n")
f_ops.write(str(num_ops)+"\n")
for op in effective_ops:
    f_ops.write(op)
f_ops.close()

# os.system(exe_file + " tmp/tmp > tmp/tmp_log_file_time")
os.system("nvprof --print-gpu-trace " + " --log-file tmp/tmp_nvprof " + exe_file + " tmp/tmp_2 > tmp/tmp_log_file_time_2")
f_ops = open(saved_file, 'w')
effective_ops = []
num_ops = 0
with open("tmp/tmp_log_file_time_2") as logs:
    for line in logs:
        splits = line.split("\t")
        current_op_time = float(splits[9])
        if(current_op_time>0):
            num_ops += 1
            effective_ops.append("\t".join(splits[:-1])+"\n")
f_ops.write(str(num_ops)+"\n")
for op in effective_ops:
    f_ops.write(op)
f_ops.close()