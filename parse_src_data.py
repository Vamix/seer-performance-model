import csv
import xlwt
import xlrd
import json
import math
import sys
import numpy as np

target = sys.argv[1] 
src_dataset_path = "data/src/"
saved_path = "data/profiled/"
tmp_result_path = "data/tmp_results/"

repeat_times = 5
time_min = 1
time_max = 1000
# for Titan Xp
algo0_kernel_names = ["void cudnn::detail::"]
algo1_kernel_names = ["maxwell_scudnn_128x1", "maxwell_scudnn_128x3", "maxwell_scudnn_128x6"]
algo2_kernel_names = ["void cudnn::detail::"]
algo6_kernel_names = ["maxwell_scudnn_winog"]
algo7_kernel_names = ["maxwell_sgemm_128x64", "maxwell_sgemm_128x12", "gemmSN_NN_kerne", "gemv2N_kernel"]
# for Titan V
# algo0_kernel_names = ["void cudnn::detail::"]
# algo1_kernel_names = ["volta_scudnn_128x64_", "volta_scudnn_128x32_", "volta_scudnn_128x128"]
# algo2_kernel_names = ["void cudnn::detail::"]
# algo6_kernel_names = ["volta_scudnn_winogra"]
# algo7_kernel_names = ["volta_sgemm_128x64_n","volta_sgemm_128x128"]

all_kernel_names = algo0_kernel_names + algo1_kernel_names + algo6_kernel_names + algo7_kernel_names
target_kernel_names = [algo0_kernel_names, algo1_kernel_names, algo2_kernel_names, [], [], [], algo6_kernel_names, algo7_kernel_names]

# obtained from CUDA calculator
blk_max_algo0_32x32 = 240
blk_max_algo0_64x128 = 120
blk_max_algo0_128x128 = 120

blk_max_algo1_128x32 = 150
blk_max_algo1_128x64 = 120
blk_max_algo1_128x128 = 60

blk_max_algo2_32x32 = 480
blk_max_algo2_64x128 = 60
blk_max_algo2_128x128 = 60

blk_max_algo6 = 60

blk_max_algo7_128x64 = 120
blk_max_algo7_128x128 = 60

def write_to_xls_seer(data, file_name, blk_max):
    # save to xlsx files
    workbook = xlwt.Workbook()
    sheet_train = workbook.add_sheet('train')
    sheet_test = workbook.add_sheet('test')
    sheet_train.write(0, 0, 'batch_size')
    sheet_train.write(0, 1, 'in_chan')
    sheet_train.write(0, 2, 'in_wid')
    sheet_train.write(0, 3, 'out_chan')
    sheet_train.write(0, 4, 'out_wid')
    sheet_train.write(0, 5, 'ker_wid')   
    sheet_train.write(0, 6, 'stride')             
    sheet_train.write(0, 7, 'padding')
    sheet_train.write(0, 8, 'algo')
    sheet_train.write(0, 9, 'time')
    sheet_train.write(0, 10, 'inst_fp_32')
    sheet_train.write(0, 11, 'dram_read_transactions')
    sheet_train.write(0, 12, 'dram_write_transactions')
    sheet_train.write(0, 13, '#block-max')
    sheet_train.write(0, 14, '#block-launch')
    sheet_train.write(0, 15, '#iter')
    sheet_train.write(0, 16, 'single_precision_fu_utilization')
    sheet_train.write(0, 17, 'dram_utilization')
    sheet_train.write(0, 18, 'type')    

    sheet_test.write(0, 0, 'batch_size')
    sheet_test.write(0, 1, 'in_chan')
    sheet_test.write(0, 2, 'in_wid')
    sheet_test.write(0, 3, 'out_chan')
    sheet_test.write(0, 4, 'out_wid')
    sheet_test.write(0, 5, 'ker_wid')   
    sheet_test.write(0, 6, 'stride')             
    sheet_test.write(0, 7, 'padding')
    sheet_test.write(0, 8, 'algo')
    sheet_test.write(0, 9, 'time')
    sheet_test.write(0, 10, 'inst_fp_32')
    sheet_test.write(0, 11, 'dram_read_transactions')
    sheet_test.write(0, 12, 'dram_write_transactions')
    sheet_test.write(0, 13, '#block-max')
    sheet_test.write(0, 14, '#block-launch')
    sheet_test.write(0, 15, '#iter')
    sheet_test.write(0, 16, 'single_precision_fu_utilization')
    sheet_test.write(0, 17, 'dram_utilization')
    sheet_test.write(0, 18, 'type')   

    index_training_set = 1
    index_test_set = 1

    for index in range(len(data)):
        if index % 10 < 7: # training set
            sheet_train.write(index_training_set, 0, int(data[index][0]))
            sheet_train.write(index_training_set, 1, int(data[index][1]))
            sheet_train.write(index_training_set, 2, int(data[index][2]))
            sheet_train.write(index_training_set, 3, int(data[index][3]))
            sheet_train.write(index_training_set, 4, int(data[index][4]))
            sheet_train.write(index_training_set, 5, int(data[index][5]))
            sheet_train.write(index_training_set, 6, int(data[index][6]))        
            sheet_train.write(index_training_set, 7, int(data[index][7]))
            sheet_train.write(index_training_set, 8, int(data[index][8]))
            sheet_train.write(index_training_set, 9, float(data[index][9]))
            sheet_train.write(index_training_set, 10, float(data[index][13]))
            sheet_train.write(index_training_set, 11, float(data[index][14]))
            sheet_train.write(index_training_set, 12, float(data[index][15]))
            sheet_train.write(index_training_set, 13, blk_max)
            sheet_train.write(index_training_set, 14, int(data[index][10]))
            sheet_train.write(index_training_set, 15, math.ceil(int(data[index][10])/blk_max))
            sheet_train.write(index_training_set, 16, int(data[index][16]))
            sheet_train.write(index_training_set, 17, int(data[index][17]))
            sheet_train.write(index_training_set, 18, -1)   
            index_training_set += 1
        else:
            sheet_test.write(index_test_set, 0, int(data[index][0]))
            sheet_test.write(index_test_set, 1, int(data[index][1]))
            sheet_test.write(index_test_set, 2, int(data[index][2]))
            sheet_test.write(index_test_set, 3, int(data[index][3]))
            sheet_test.write(index_test_set, 4, int(data[index][4]))
            sheet_test.write(index_test_set, 5, int(data[index][5]))
            sheet_test.write(index_test_set, 6, int(data[index][6]))       
            sheet_test.write(index_test_set, 7, int(data[index][7]))
            sheet_test.write(index_test_set, 8, int(data[index][8]))
            sheet_test.write(index_test_set, 9, float(data[index][9]))
            sheet_test.write(index_test_set, 10, float(data[index][13]))
            sheet_test.write(index_test_set, 11, float(data[index][14]))
            sheet_test.write(index_test_set, 12, float(data[index][15]))
            sheet_test.write(index_test_set, 13, blk_max)
            sheet_test.write(index_test_set, 14, int(data[index][10]))
            sheet_test.write(index_test_set, 15, math.ceil(int(data[index][10])/blk_max))
            sheet_test.write(index_test_set, 16, int(data[index][16]))
            sheet_test.write(index_test_set, 17, int(data[index][17]))
            sheet_test.write(index_test_set, 18, -1)   
            index_test_set += 1     

    workbook.save(file_name)

def write_to_xls(data, file_name, metric_name="time"):
    # save to xlsx files
    workbook = xlwt.Workbook()
    sheet_train = workbook.add_sheet('train')
    sheet_test = workbook.add_sheet('test')
    sheet_train.write(0, 0, 'batch_size')
    sheet_train.write(0, 1, 'in_chan')
    sheet_train.write(0, 2, 'in_wid')
    sheet_train.write(0, 3, 'out_chan')
    sheet_train.write(0, 4, 'out_wid')
    sheet_train.write(0, 5, 'ker_wid')   
    sheet_train.write(0, 6, 'stride')             
    sheet_train.write(0, 7, 'padding')
    sheet_train.write(0, 8, 'algo')
    sheet_train.write(0, 9, 'time')

    sheet_test.write(0, 0, 'batch_size')
    sheet_test.write(0, 1, 'in_chan')
    sheet_test.write(0, 2, 'in_wid')
    sheet_test.write(0, 3, 'out_chan')
    sheet_test.write(0, 4, 'out_wid')
    sheet_test.write(0, 5, 'ker_wid')   
    sheet_test.write(0, 6, 'stride')             
    sheet_test.write(0, 7, 'padding')
    sheet_test.write(0, 8, 'algo')
    sheet_test.write(0, 9, metric_name)

    index_training_set = 1
    index_test_set = 1

    for index in range(len(data)):
        if index % 10 < 7: # training set
            sheet_train.write(index_training_set, 0, int(data[index][0]))
            sheet_train.write(index_training_set, 1, int(data[index][1]))
            sheet_train.write(index_training_set, 2, int(data[index][2]))
            sheet_train.write(index_training_set, 3, int(data[index][3]))
            sheet_train.write(index_training_set, 4, int(data[index][4]))
            sheet_train.write(index_training_set, 5, int(data[index][5]))
            sheet_train.write(index_training_set, 6, int(data[index][6]))        
            sheet_train.write(index_training_set, 7, int(data[index][7]))
            sheet_train.write(index_training_set, 8, int(data[index][8]))
            sheet_train.write(index_training_set, 9, float(data[index][9]))
            index_training_set += 1
        else:
            sheet_test.write(index_test_set, 0, int(data[index][0]))
            sheet_test.write(index_test_set, 1, int(data[index][1]))
            sheet_test.write(index_test_set, 2, int(data[index][2]))
            sheet_test.write(index_test_set, 3, int(data[index][3]))
            sheet_test.write(index_test_set, 4, int(data[index][4]))
            sheet_test.write(index_test_set, 5, int(data[index][5]))
            sheet_test.write(index_test_set, 6, int(data[index][6]))       
            sheet_test.write(index_test_set, 7, int(data[index][7]))
            sheet_test.write(index_test_set, 8, int(data[index][8]))
            sheet_test.write(index_test_set, 9, float(data[index][9]))
            index_test_set += 1     
    workbook.save(file_name)

def write_to_xls_with_all_algos(data, file_name):
    # save to xlsx files
    workbook = xlwt.Workbook()
    sheet_test = workbook.add_sheet('test')
    sheet_test.write(0, 0, 'batch_size')
    sheet_test.write(0, 1, 'in_chan')
    sheet_test.write(0, 2, 'in_wid')
    sheet_test.write(0, 3, 'out_chan')
    sheet_test.write(0, 4, 'out_wid')
    sheet_test.write(0, 5, 'ker_wid')   
    sheet_test.write(0, 6, 'stride')             
    sheet_test.write(0, 7, 'padding')
    sheet_test.write(0, 8, 'time_algo0')
    sheet_test.write(0, 9, 'time_algo1')
    sheet_test.write(0, 10, 'time_algo2')
    sheet_test.write(0, 11, 'time_algo3')
    sheet_test.write(0, 12, 'time_algo4')
    sheet_test.write(0, 13, 'time_algo5')
    sheet_test.write(0, 14, 'time_algo6')
    sheet_test.write(0, 15, 'time_algo7')    
    sheet_test.write(0, 16, 'best_real')    
    sheet_test.write(0, 17, 'best_cudnn')        

    index_test_set = 1

    for index in range(len(data)):
        sheet_test.write(index_test_set, 0, int(data[index][0]))
        sheet_test.write(index_test_set, 1, int(data[index][1]))
        sheet_test.write(index_test_set, 2, int(data[index][2]))
        sheet_test.write(index_test_set, 3, int(data[index][3]))
        sheet_test.write(index_test_set, 4, int(data[index][4]))
        sheet_test.write(index_test_set, 5, int(data[index][5]))
        sheet_test.write(index_test_set, 6, int(data[index][6]))       
        sheet_test.write(index_test_set, 7, int(data[index][7]))
        sheet_test.write(index_test_set, 8, float(data[index][8]))
        sheet_test.write(index_test_set, 9, float(data[index][9]))
        sheet_test.write(index_test_set, 10, float(data[index][10]))
        sheet_test.write(index_test_set, 11, float(data[index][11]))
        sheet_test.write(index_test_set, 12, float(data[index][12]))
        sheet_test.write(index_test_set, 13, float(data[index][13]))
        sheet_test.write(index_test_set, 14, float(data[index][14]))
        sheet_test.write(index_test_set, 15, float(data[index][15]))    
        sheet_test.write(index_test_set, 16, int(data[index][16]))       
        sheet_test.write(index_test_set, 17, int(data[index][17]))
        index_test_set += 1     
    workbook.save(file_name)

def this_line_contains_op_of(line, algo):
    global target_kernel_names
    for kernel in target_kernel_names[algo]:
        if kernel in line:
            return kernel
    return None

def parse_nvprof_data(algo_name):
    num_ops = 0
    ops_lines = []
    with open(src_dataset_path + 'prof-ops-' + algo_name + '.txt') as tmp_log:
        ops_lines = tmp_log.readlines()
        num_ops = int(ops_lines[0])

    global all_kernel_names
    nvprof_trace_lines = []
    for i in range(repeat_times):
        nvprof_trace_lines.append([])
        with open(tmp_result_path + "nvprof_trace_" + algo_name + "_" + str(i)) as logs:
            tmp_lines = logs.readlines()
            for index in range(len(tmp_lines)):
                for kernel in all_kernel_names:
                    if(kernel in tmp_lines[index]):
                        nvprof_trace_lines[i].append(tmp_lines[index])    

    nvprof_metrics_lines = []
    with open(tmp_result_path + "nvprof_metrics_" + algo_name) as logs:
        tmp_lines = logs.readlines()
        for index in range(len(tmp_lines)):
            for kernel in all_kernel_names:
                if(kernel in tmp_lines[index]):
                    nvprof_metrics_lines.append(tmp_lines[index])    

    averaged_time_lines = []
    for i in range(num_ops):
        tmp_time = []
        for j in range(repeat_times):
            # print("[DEBUG], j = ", j, " i = ", i)
            time_str = nvprof_trace_lines[j][i].split("  ")[1]
            if("ms" in time_str):
                time_one_try = float(time_str.split("ms")[0])
            elif("us" in time_str):
                time_one_try = float(time_str.split("us")[0])/1000
            else: # "s"
                time_one_try = float(time_str.split("s")[0])*1000
            tmp_time.append(time_one_try)
        current_op_time = np.mean(tmp_time)
        averaged_time_lines.append(ops_lines[i+1].split("\n")[0] + "\t" + str(current_op_time))

    if(len(nvprof_trace_lines[0])!= num_ops):
        print("#nvprof trace result[" + str(len(nvprof_trace_lines[0])) + "] not match #ops in src_file[" + str(num_ops) + "]")
    elif (len(nvprof_metrics_lines)!= num_ops):
        print("#nvprof metrics result[" + str(len(nvprof_metrics_lines)) + "] not match #ops in src_file[" + str(num_ops) + "]")
    else:
        f_result = open(saved_path + 'Test-set-I-' + algo_name + '.csv','w')
        f_csv = csv.writer(f_result)
        reult_title = ['batch_size', 'in_chan', 'in_wid', 'out_chan', 'out_wid', 'ker_wid', 'stride', 'padding', 'algo', 'time', '#block-launch', '#warp-per-blk', '#warp', 'inst_fp32','dram_read_transactions', 'dram_write_transactions', 'single_precision_fu_utilization', 'dram_utilization', 'kernel_name']

        f_csv.writerow(reult_title)
        for i in range(len(nvprof_metrics_lines)):
            one_result_line = averaged_time_lines[i].split("\n")[0]
            one_result_line = one_result_line.split("\t")
            algo = int(one_result_line[8])
            kernel_name_in_trace_line = this_line_contains_op_of(nvprof_trace_lines[0][i], algo)
            kernel_name_in_metric_line = this_line_contains_op_of(nvprof_metrics_lines[i], algo)
            if kernel_name_in_trace_line is not None:
                grid = list(map(eval, nvprof_trace_lines[0][i].split("(")[1].split(")")[0].split(" ")))
                block = list(map(eval, nvprof_trace_lines[0][i].split("(")[2].split(")")[0].split(" ")))
                num_block = grid[0] * grid[1] * grid[2]
                num_warp_per_block = int(block[0] * block[1] * block[2] / 32)
                num_warp = num_block * num_warp_per_block
                one_result_line += [num_block, num_warp_per_block, num_warp]
                kernel_full_name = nvprof_trace_lines[0][i].split(kernel_name_in_trace_line[0:5])[1].split("[")[0]
            if kernel_name_in_metric_line is not None:
                tmp_splits = nvprof_metrics_lines[i].split(kernel_name_in_metric_line)[1].split("\n")[0].split(" ")
                metrics = []
                for j in range(len(tmp_splits)):
                    if tmp_splits[j] != '':
                        metrics.append(tmp_splits[j])
                metrics.pop(3)
                metrics.pop(4)
                metrics[3] = metrics[3].split("(")[1].split(")")[0]
                metrics[4] = metrics[4].split("(")[1].split(")")[0]
                one_result_line += metrics
            one_result_line.append(kernel_full_name)   
            f_csv.writerow(one_result_line)   

        f_result.close()

def parse_profiler_data(algo_name):
    num_ops = 0
    ops_lines = []
    with open(src_dataset_path + 'prof-ops-' + algo_name + '.txt') as tmp_log:
        ops_lines = tmp_log.readlines()
        num_ops = int(ops_lines[0])
    
    time_results_lines = []
    for i in range(repeat_times):
        time_results_lines.append([])
        with open(tmp_result_path + 'time_results_' + algo_name + '_' + str(i)) as logs:
            tmp_lines = logs.readlines()
            for index in range(len(tmp_lines)):
                time_results_lines[i].append(tmp_lines[index])    

    check_flag = True
    for i in range(repeat_times - 1):
        if(len(time_results_lines[i]) != len(time_results_lines[i+1])):
            check_flag = False
            print("#results not match across different tries.")

    if check_flag:
        if(len(time_results_lines[0])!= num_ops):
            print("#result[" + str(len(time_results_lines[0])) + "] not match #ops in src_file[" + str(num_ops) + "]")   
        else:
            f_result = open(saved_path + 'Test-set-I-' + algo_name + '.csv','w')  # save final average time
            f_csv = csv.writer(f_result)
            reult_title = ['batch_size', 'in_chan', 'in_wid', 'out_chan', 'out_wid', 'ker_wid', 'stride', 'padding', 'algo', 'time']
            f_csv.writerow(reult_title)
            for i in range(num_ops):
                tmp_time = []
                for j in range(1, repeat_times):
                    time_str = time_results_lines[j][i].split("\t")[9]
                    tmp_time.append(float(time_str))
                current_op_time = np.mean(tmp_time)
                one_result_line = ops_lines[i+1].split("\n")[0].split("\t")
                one_result_line.append(current_op_time)
                f_csv.writerow(one_result_line)  
            f_result.close()


if target == 'algo0':
    parse_nvprof_data('algo0')
    # Algo0, split src data into three categories: 32x32, 64x128, 128x128
    result_file = saved_path + "Test-set-I-algo0.csv"  
    data = []
    with open(result_file) as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            data.append(row)
        data_32x32 = []
        data_64x128 = []
        data_128x128 = []
        for i in range(1, len(data)):
            data[i][9] = float(data[i][9])
            warp_per_blk = int(data[i][11])
            if warp_per_blk == 2:
                data_32x32.append(data[i])
            else:
                batch_size = int(data[i][0])
                out_wid = int(data[i][4])
                out_chan = int(data[i][3])
                num_blk_launch = int(data[i][10])
                num_blk_64x128 = math.ceil(batch_size * out_wid * out_wid /64) * math.ceil(out_chan/128)
                num_blk_128x128 = math.ceil(batch_size * out_wid * out_wid /128) * math.ceil(out_chan/128)
                if num_blk_64x128 == num_blk_launch and num_blk_128x128 == num_blk_launch:
                    print("==WARNING== op may be 64x128 or 128x128: ", data[i])
                    break
                elif num_blk_64x128 == num_blk_launch:
                    data_64x128.append(data[i])
                elif num_blk_128x128 == num_blk_launch:
                    data_128x128.append(data[i])

    data_32x32 = sorted(data_32x32,key=(lambda x:x[9]))
    data_64x128 = sorted(data_64x128,key=(lambda x:x[9]))
    data_128x128 = sorted(data_128x128,key=(lambda x:x[9]))

    data_32x32_new = []
    for i in range(1, len(data_32x32)):
        if data_32x32[i][9] > time_min and data_32x32[i][9] < time_max:
            data_32x32_new.append(data_32x32[i])
    data_64x128_new = []
    for i in range(1, len(data_64x128)):
        if data_64x128[i][9] > time_min and data_64x128[i][9] < time_max:
            data_64x128_new.append(data_64x128[i])
    data_128x128_new = []
    for i in range(1, len(data_128x128)):
        if data_128x128[i][9] > time_min and data_128x128[i][9] < time_max:
            data_128x128_new.append(data_128x128[i])

    write_to_xls_seer(data_32x32_new, saved_path + "algo0_32x32.xls", blk_max_algo0_32x32)
    write_to_xls_seer(data_64x128_new, saved_path + "algo0_64x128.xls", blk_max_algo0_64x128)
    write_to_xls_seer(data_128x128_new, saved_path + "algo0_128x128.xls", blk_max_algo0_128x128)

elif target == 'algo1':
    parse_nvprof_data('algo1')
    # Algo1, split src data into three categories: 128x32, 128x64, 128x128
    result_file = saved_path + "Test-set-I-algo1.csv"  
    data = []
    with open(result_file) as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            data.append(row)
        data_128x32 = []
        data_128x64 = []
        data_128x128 = []
        for i in range(1, len(data)):
            data[i][9] = float(data[i][9])
            kernel_name = data[i][18]
            if "128x1" in kernel_name:
                data_128x128.append(data[i])
            elif "128x3" in kernel_name:
                data_128x32.append(data[i])
            elif "128x6" in kernel_name:
                data_128x64.append(data[i])

    data_128x32 = sorted(data_128x32,key=(lambda x:x[9]))
    data_128x64 = sorted(data_128x64,key=(lambda x:x[9]))
    data_128x128 = sorted(data_128x128,key=(lambda x:x[9]))

    data_128x32_new = []
    for i in range(1, len(data_128x32)):
        if data_128x32[i][9] > time_min and data_128x32[i][9] < time_max:
            data_128x32_new.append(data_128x32[i])
    data_128x64_new = []
    for i in range(1, len(data_128x64)):
        if data_128x64[i][9] > time_min and data_128x64[i][9] < time_max:
            data_128x64_new.append(data_128x64[i])
    data_128x128_new = []
    for i in range(1, len(data_128x128)):
        if data_128x128[i][9] > time_min and data_128x128[i][9] < time_max:
            data_128x128_new.append(data_128x128[i])



    write_to_xls_seer(data_128x32_new, saved_path + "algo1_128x32.xls", blk_max_algo1_128x32)
    write_to_xls_seer(data_128x64_new, saved_path + "algo1_128x64.xls", blk_max_algo1_128x64)
    write_to_xls_seer(data_128x128_new, saved_path + "algo1_128x128.xls", blk_max_algo1_128x128)

elif target == 'algo2':
    parse_nvprof_data('algo2')
    # Algo2, split src data into three categories: 32x32, 64x128, 128x128
    result_file = saved_path + "Test-set-I-algo2.csv"  
    data = []
    with open(result_file) as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            data.append(row)
        data_32x32 = []
        data_64x128 = []
        data_128x128 = []
        for i in range(1, len(data)):
            data[i][9] = float(data[i][9])
            warp_per_blk = int(data[i][11])
            if warp_per_blk == 2:
                data_32x32.append(data[i])
            else:
                batch_size = int(data[i][0])
                out_wid = int(data[i][4])
                out_chan = int(data[i][3])
                num_blk_launch = int(data[i][10])
                num_blk_64x128 = math.ceil(batch_size * out_wid * out_wid /64) * math.ceil(out_chan/128)
                num_blk_128x128 = math.ceil(batch_size * out_wid * out_wid /128) * math.ceil(out_chan/128)
                if num_blk_64x128 == num_blk_launch and num_blk_128x128 == num_blk_launch:
                    print("==WARNING== op may be 64x128 or 128x128: ", data[i])
                    break
                elif num_blk_64x128 == num_blk_launch:
                    data_64x128.append(data[i])
                elif num_blk_128x128 == num_blk_launch:
                    data_128x128.append(data[i])

    data_32x32 = sorted(data_32x32,key=(lambda x:x[9]))
    data_64x128 = sorted(data_64x128,key=(lambda x:x[9]))
    data_128x128 = sorted(data_128x128,key=(lambda x:x[9]))

    data_32x32_new = []
    for i in range(1, len(data_32x32)):
        if data_32x32[i][9] > time_min and data_32x32[i][9] < time_max:
            data_32x32_new.append(data_32x32[i])
    data_64x128_new = []
    for i in range(1, len(data_64x128)):
        if data_64x128[i][9] > time_min and data_64x128[i][9] < time_max:
            data_64x128_new.append(data_64x128[i])
    data_128x128_new = []
    for i in range(1, len(data_128x128)):
        if data_128x128[i][9] > time_min and data_128x128[i][9] < time_max:
            data_128x128_new.append(data_128x128[i])

    write_to_xls_seer(data_32x32_new, saved_path + "algo2_32x32.xls", blk_max_algo2_32x32)
    write_to_xls_seer(data_64x128_new, saved_path + "algo2_64x128.xls", blk_max_algo2_64x128)
    write_to_xls_seer(data_128x128_new, saved_path + "algo2_128x128.xls", blk_max_algo2_128x128)

elif target == 'algo4':
    parse_profiler_data('algo4')
    # Algo4, split src data into categories: 16, 32, 64, 128, 256
    result_file = saved_path + "Test-set-I-algo4.csv"  
    data = []
    with open(result_file) as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            data.append(row)
        data_16 = []
        data_32 = []
        data_64 = []
        data_128 = [] 
        data_256 = []        
        for i in range(1, len(data)):
            in_wid = int(data[i][2])
            if in_wid <=16:
                data_16.append(data[i])
            elif in_wid <= 32:
                data_32.append(data[i])
            elif in_wid <= 64:
                data_64.append(data[i])
            elif in_wid <= 128:
                data_128.append(data[i])
            elif in_wid <= 256:
                data_256.append(data[i])

    write_to_xls(data_16, saved_path + "algo4_16.xls")
    write_to_xls(data_32, saved_path + "algo4_32.xls")
    write_to_xls(data_64, saved_path + "algo4_64.xls")
    write_to_xls(data_128, saved_path + "algo4_128.xls")
    write_to_xls(data_256, saved_path + "algo4_256.xls")

elif target == 'algo5':
    parse_profiler_data('algo5')
    # Algo5, split src data into categories: 16, 32, 64, 128, 256
    result_file = saved_path + "Test-set-I-algo5.csv"  
    data = []
    with open(result_file) as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            data.append(row)
        data_new = []
        for i in range(1, len(data)):
            data_new.append(data[i])
    write_to_xls(data_new, saved_path + "algo5.xls")

elif target == 'algo6':
    parse_nvprof_data('algo6')
    # Algo6, split src data into three categories: 32x32, 64x128, 128x128
    result_file = saved_path + "Test-set-I-algo6.csv"  
    data = []
    with open(result_file) as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            data.append(row)
        data_148 = []
        data_228 = []
        data_418 = []
        for i in range(1, len(data)):
            kernel_name = data[i][18]
            if "148" in kernel_name:
                data_148.append(data[i])
            elif "418" in kernel_name:
                data_418.append(data[i])
            elif "228" in kernel_name:
                data_228.append(data[i])

    write_to_xls_seer(data_148, saved_path + "algo6_148.xls", blk_max_algo6)
    write_to_xls_seer(data_418, saved_path + "algo6_418.xls", blk_max_algo6)
    write_to_xls_seer(data_228, saved_path + "algo6_228.xls", blk_max_algo6)

elif target == 'algo7':
    parse_nvprof_data('algo7')
    # Algo7, split src data into three categories: 32x32, 64x128, 128x128
    result_file = saved_path + "Test-set-I-algo7.csv"  
    data = []
    with open(result_file) as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            data.append(row)
        data_sn = []
        data_128x64_5x5 = []
        data_128x64_3x3 = []
        data_128x128_5x5 = []
        data_128x128_3x3 = []
        for i in range(1, len(data)):
            kernel_name = data[i][18]
            ker_wid = int(data[i][5])
            if "gemmSN_NN_kerne" in kernel_name or "gemv2N_kernel" in kernel_name:
                data_sn.append(data[i])
            elif "128x64" in kernel_name:
                if ker_wid == 3:
                    data_128x64_3x3.append(data[i])
                elif ker_wid == 5:
                    data_128x64_5x5.append(data[i])
            elif "128x12" in kernel_name:
                if ker_wid == 3:
                    data_128x128_3x3.append(data[i])
                elif ker_wid == 5:
                    data_128x128_5x5.append(data[i])            

    write_to_xls_seer(data_128x64_5x5, saved_path + "algo7_128x64_5x5.xls", blk_max_algo7_128x64)
    write_to_xls_seer(data_128x64_3x3, saved_path + "algo7_128x64_3x3.xls", blk_max_algo7_128x64)
    write_to_xls_seer(data_128x128_5x5, saved_path + "algo7_128x128_5x5.xls", blk_max_algo7_128x128)
    write_to_xls_seer(data_128x128_3x3, saved_path + "algo7_128x128_3x3.xls", blk_max_algo7_128x128)
    # write_to_xls_seer(data_sn, saved_path + "algo7_sn.xls", blk_max_algo7_128x128)

elif target == 'Test-set-II': 
    result_lines = []
    with open(tmp_result_path + target) as tmp_log:
        result_lines = tmp_log.readlines()

    result_lines_new = []
    for i in range(1, len(result_lines)):
        result_lines_new.append(result_lines[i].split("\n")[0].split("\t"))
    
    write_to_xls_with_all_algos(result_lines_new, saved_path + target + ".xls")

elif target == 'Algo-pick-set': 
    result_lines = []
    with open(tmp_result_path + target) as tmp_log:
        result_lines = tmp_log.readlines()

    result_lines_new = []
    for i in range(1, len(result_lines)):
        op_config = result_lines[i].split("\n")[0].split("\t")
        current_kernel_time = 1000000
        for algo_index in range(8+0, 8+8):
            tmp_time = float(op_config[algo_index])
            if tmp_time > 0 and tmp_time < current_kernel_time:
                current_kernel_time = tmp_time
        if current_kernel_time > time_min:
            result_lines_new.append(op_config)
    
    write_to_xls_with_all_algos(result_lines_new, saved_path + target + ".xls")

elif target == 'algo2-pre-kernel':
    num_ops = 0
    ops_lines = []
    with open(src_dataset_path + 'prof-ops-algo2.txt') as tmp_log:
        ops_lines = tmp_log.readlines()
        num_ops = int(ops_lines[0])

    nvprof_trace_lines = []
    pre_kernel_names = ["im2col4d_kernel"]
    for i in range(repeat_times):
        nvprof_trace_lines.append([])
        with open(tmp_result_path + "nvprof_trace_algo2_" + str(i)) as logs:
            tmp_lines = logs.readlines()
            for index in range(len(tmp_lines)):
                for kernel in pre_kernel_names:
                    if(kernel in tmp_lines[index]):
                        nvprof_trace_lines[i].append(tmp_lines[index])    

    averaged_time_lines = []
    for i in range(num_ops):
        tmp_time = []
        for j in range(1, repeat_times):
            time_str = nvprof_trace_lines[j][i].split("  ")[1]
            if("ms" in time_str):
                time_one_try = float(time_str.split("ms")[0])
            elif("us" in time_str):
                time_one_try = float(time_str.split("us")[0])/1000
            else: # "s"
                time_one_try = float(time_str.split("s")[0])*1000
            tmp_time.append(time_one_try)
        current_op_time = np.mean(tmp_time)
        op_config = ops_lines[i+1].split("\n")[0].split("\t")
        op_config.append(current_op_time)
        averaged_time_lines.append(op_config)    
    write_to_xls(averaged_time_lines, saved_path + "algo2_pre.xls")

elif target == 'algo7-pre-kernel':
    num_ops = 0
    ops_lines = []
    with open(src_dataset_path + 'prof-ops-algo7.txt') as tmp_log:
        ops_lines = tmp_log.readlines()
        num_ops = int(ops_lines[0])

    nvprof_trace_lines = []
    pre_kernel_names = ["winogradForwardData4x4", "winogradForwardData9x9_5x5"]
    for i in range(3):
        nvprof_trace_lines.append([])

    kernel_type = []
    for i in range(repeat_times):
        for index_pre_kernel in range(3):
            nvprof_trace_lines[index_pre_kernel].append([])
        with open(tmp_result_path + "nvprof_trace_algo7_" + str(i)) as logs:
            tmp_lines = logs.readlines()
            for index in range(len(tmp_lines)):
                for kernel in pre_kernel_names:
                    if(kernel in tmp_lines[index]):
                        nvprof_trace_lines[0][i].append(tmp_lines[index])    # winogradForwardData
                        nvprof_trace_lines[1][i].append(tmp_lines[index])    # winogradForwardFilter
                        nvprof_trace_lines[2][i].append(tmp_lines[index])    # winogradForwardOutput
                        if "maxwell" in tmp_lines[index+2]:
                            kernel_type.append(tmp_lines[index+2].split("maxwell")[1].split("[")[0])  
                        else:
                            kernel_type.append("sn")

    time_128x128_3x3_pre_kernel = []
    time_128x128_5x5_pre_kernel = []
    time_128x64_3x3_pre_kernel = []
    time_128x64_5x5_pre_kernel = []

    for i in range(num_ops):
        op_config = ops_lines[i+1].split("\n")[0].split("\t")
        pre_kernel_time_list = []
        for op_index in range(3):
            tmp_time = []
            for j in range(1, repeat_times):
                time_str = nvprof_trace_lines[op_index][j][i].split("  ")[1]
                if("ms" in time_str):
                    time_one_try = float(time_str.split("ms")[0])
                elif("us" in time_str):
                    time_one_try = float(time_str.split("us")[0])/1000
                else: # "s"
                    time_one_try = float(time_str.split("s")[0])*1000
                tmp_time.append(time_one_try)
            pre_kernel_time_list.append(np.mean(tmp_time))
        op_config.append(np.sum(pre_kernel_time_list))
        if "128x64" in kernel_type[i]:
            if int(op_config[5]) == 3:
                time_128x64_3x3_pre_kernel.append(op_config)
            elif int(op_config[5]) == 5:
                time_128x64_5x5_pre_kernel.append(op_config)
            else:
                print("Error: kernel width is not supportted. ")
        elif "128x128" in kernel_type[i]:
            if int(op_config[5]) == 3:
                time_128x128_3x3_pre_kernel.append(op_config)
            elif int(op_config[5]) == 5:
                time_128x128_5x5_pre_kernel.append(op_config)
            else:
                print("Error: kernel width is not supportted. ")            
    write_to_xls(time_128x64_3x3_pre_kernel, saved_path + "algo7_128x64_3x3_pre.xls")   
    write_to_xls(time_128x64_5x5_pre_kernel, saved_path + "algo7_128x64_5x5_pre.xls")   
    write_to_xls(time_128x128_3x3_pre_kernel, saved_path + "algo7_128x128_3x3_pre.xls")   
    write_to_xls(time_128x128_5x5_pre_kernel, saved_path + "algo7_128x128_5x5_pre.xls")    

elif target == 'algo5-ctree':
    num_ops = 0
    ops_lines = []
    with open(src_dataset_path + 'prof-ops-algo5-ctree.txt') as tmp_log:
        ops_lines = tmp_log.readlines()
        num_ops = int(ops_lines[0])

    nvprof_metric_lines = []
    with open(tmp_result_path + "nvprof_metrics_algo5-ctree") as logs:
        tmp_lines = logs.readlines()
        for index in range(len(tmp_lines)):
            if "gemm" in tmp_lines[index]:
                nvprof_metric_lines.append(tmp_lines[index])    

    if len(nvprof_metric_lines) != num_ops:
        print("Error: # of ops not match # of profiling results")

    results = []
    for i in range(num_ops):
        fp_utilization = int(nvprof_metric_lines[i].split("(")[2].split(")")[0])
        dram_utilization = int(nvprof_metric_lines[i].split("(")[3].split(")")[0])
        if fp_utilization >= 8:
            kernel_type = 0
        elif dram_utilization >= 8:
            kernel_type = 1
        else:
            kernel_type = 2
        op_config = ops_lines[i+1].split("\n")[0].split("\t")
        op_config.append(kernel_type)
        results.append(op_config)    
    write_to_xls(results, saved_path + "algo5-ctree.xls", metric_name="kernel_type")    

elif target == 'layout-trans':
    time_results = []
    op_list = []    
    target_kernel_list = ["SwapDimension1And2InTensor3UsingTiles"]
    with open(tmp_result_path + 'layout') as logs:
        tmp_lines = logs.readlines()
        for index in range(len(tmp_lines)):
            for target_kernel in target_kernel_list:
                if(target_kernel in tmp_lines[index]):
                    time_str = tmp_lines[index].split("s")[1]
                    if("m" in time_str):
                        time_value = float(time_str.split("m")[0])
                    elif("u" in time_str):
                        time_value = float(time_str.split("u")[0])/1000
                    elif("n" in time_str):
                        time_value = float(time_str.split("n")[0])/1000000
                    else:
                        time_value = float(time_str)*1000
                    time_results.append(time_value)
    with open(tmp_result_path + 'layout_ops') as logs:
        op_list = logs.readlines()
    if (len(time_results)//2 != len(op_list)):
        print("[ERROR] # of ops not match # of results")
    else:
        results = []
        for i in range(len(op_list)):
            op_config = op_list[i].split("\n")[0].split("\t")
            op_config.append(time_results[i*2])
            results.append(op_config)
        write_to_xls(results, saved_path + "data_layout.xls")    
