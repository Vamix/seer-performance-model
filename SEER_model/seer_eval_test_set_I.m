
%% read source data
% file format: 
%   1      2      3       4       5       6      7     8   9    10  
% batch in_chan in_wid out_chan out_wid kernel stride pad algo time 
%   11		   12    	  13   	 	  14     	    15        16      17      18       19
% inst_fp  dram_read  dram_write  #block-max  #block_launch  #iter  fp_uti  dram_uti  type

% to be computed in step0: 
%      20            21             22                23              24
% inst_fp_blk  dram_read_blk  dram_write_blk   dram_read_ratio  dram_write_ratio
clear;
device = 'TitanXp';
kernel_names = ["algo0_128x128", "algo0_32x32", "algo0_64x128", ... 
                "algo1_128x128", "algo1_128x32", "algo1_128x64", ... 
                "algo2_128x128", "algo2_32x32", "algo2_64x128", ... 
                "algo6_148", "algo6_418", "algo6_228", ... 
                "algo7_128x128_3x3", "algo7_128x128_5x5", "algo7_128x64_3x3", "algo7_128x64_5x5"];

[~, num_kernels] = size(kernel_names);

dataset_path = ['../data/SEER/pre_collected_data_', device, '/'];
save_path = ['saved_model_',device,'/'];
suffix = '.xls';
predict_time_save_path = ['saved_predicted_time_',device,'/'];

%% evaluate
for index = 1: num_kernels
    kernel_name = kernel_names{index};
    file_name = [dataset_path, kernel_name,suffix];
    [train_data,~]=xlsread(file_name,'train');
    [test_data,~]=xlsread(file_name,'test');    
%% STEP0: Data preprocess
    [train_set_size,~] = size(train_data);
    [test_set_size,~] = size(test_data);

    if(strncmp(kernel_name,'algo0',5) || strncmp(kernel_name,'algo1',5) || strncmp(kernel_name,'algo2',5))
        algo_class = 'gemm'; 
    else
        if(strncmp(kernel_name,'algo6',5) || strncmp(kernel_name,'algo7',5))
            algo_class = 'winograd'; 
        end
    end

    % This can improve dram regression tree accuracy.
    if(strncmp(kernel_name,'algo1_128x32',12))
        for i = 1:test_set_size
            if(test_data(i,4) < 32)
                test_data(i,4) = 32;
            end
        end
    end
    
    test_data(:,20) = test_data(:,11)./test_data(:,15);
    test_data(:,21) = test_data(:,12)./test_data(:,15);
    test_data(:,22) = test_data(:,13)./test_data(:,15);
    if(strcmp(algo_class,'gemm'))
        test_data(:,23) = test_data(:,21)./(test_data(:,6).^2 .* test_data(:,2));
        test_data(:,24) = test_data(:,22);
    else
        if(strcmp(algo_class,'winograd'))
            test_data(:,23) = test_data(:,21)./test_data(:,2);
            test_data(:,24) = test_data(:,22);        
        end
    end

    [row, ~] = size(test_data);
    for i = 1:row   
        if(test_data(i,17) >= 8)       % compute-bound
            test_data(i,19) = 0;
        else
            if(test_data(i,18) >= 8)   % memory-bound
                test_data(i,19) = 1;
            else                        % under-utilized
                test_data(i,19) = 2;
            end
        end
    end

    test_data(:,16) = ceil(test_data(:,15)./test_data(:,14));
    test_data_copy = test_data; % use to store predicted results

    %% Load pretrained model
    load([save_path,'ctree_',kernel_name,'.mat'],'TypeClassificationTree');
    load([save_path, 'rtree_dread_',kernel_name,'.mat'],'DramReadRegTree');
    load([save_path, 'rtree_dwrite_',kernel_name,'.mat'],'DramWriteRegTree');
    coeff_flop = csvread([save_path, 'flop_',kernel_name,'.csv']);
    coeff_compute = csvread([save_path, 'conv_',kernel_name,'_compute.csv']);
    coeff_memory = csvread([save_path, 'conv_',kernel_name,'_memory.csv']);
    coeff_under = csvread([save_path, 'conv_',kernel_name,'_under_utilized.csv']);

    %% STEP1: predict type of each kernel
    fprintf('Step1: Classification...\n');
    X_test = test_data(:, [2 3 4 5 6 7 15]);
    Y_test = test_data(:, 19);
    [predict_type,~] = predict(TypeClassificationTree, X_test);
    test_data_copy(:,19) = predict_type;

    [row, ~] = size(X_test);
    err_type = 1 - sum(predict_type == Y_test)/row;

    %% STEP2: predict [Dynamic Metrics]:# of dram transactions
    fprintf('Step2: Predicting #dram_transactions...\n');

    if(strcmp(algo_class,'gemm'))
        X_test = test_data(:, [2 3 4 5 6 7 15]);
        size_one_kernel = test_data(:,6).* test_data(:,6).* test_data(:,2);
        size_all_kernel = size_one_kernel.*test_data(:,4);
        mid_size = test_data(:,5).* test_data(:,5).* test_data(:,2).*test_data(:,6).* test_data(:,6);
        X_test = [X_test,size_one_kernel,size_all_kernel,mid_size];

        [predict_dram_read_ratio,~] = predict(DramReadRegTree,X_test);
        test_data_copy(:,23) = predict_dram_read_ratio;
        test_data_copy(:,21) = predict_dram_read_ratio .* test_data(:,6).* test_data(:,6).* test_data(:,2);

        [predict_dram_write_ratio,~] = predict(DramWriteRegTree,X_test);
        test_data_copy(:,24) = predict_dram_write_ratio;
        test_data_copy(:,22) = test_data_copy(:,24);

    else
        if(strcmp(algo_class,'winograd'))
            X_test = test_data(:, [2 3 4 5 6 7]);
            [predict_dram_read_ratio,~] = predict(DramReadRegTree,X_test);
            test_data_copy(:,23) = predict_dram_read_ratio;
            test_data_copy(:,21) = predict_dram_read_ratio .* test_data(:,2);

            [predict_dram_write_ratio,~] = predict(DramWriteRegTree,X_test);
            test_data_copy(:,24) = predict_dram_write_ratio;
            test_data_copy(:,22) = test_data_copy(:,24);  
        end
    end

    mspe_dram_read = sqrt(mean(((test_data_copy(:,21) - test_data(:,21))./test_data(:,21)) .^ 2));
    avg_dram_read = mean(abs(test_data_copy(:,21) - test_data(:,21))./test_data(:,21));
    mse_dram_read = sqrt(mean(((test_data_copy(:,21) - test_data(:,21))) .^ 2));

    mspe_dram_write = sqrt(mean(((test_data_copy(:,22) - test_data(:,22))./test_data(:,22)) .^ 2));
    avg_dram_write = mean(abs(test_data_copy(:,22) - test_data(:,22))./test_data(:,22));
    mse_dram_write = sqrt(mean(((test_data_copy(:,22) - test_data(:,22))) .^ 2));

    %% STEP3: predict [Static Metrics]: # of flop instructions
    fprintf('Step3: Predicting #flop_inst...\n');
    test_data_copy(:,20) = 1;
    Y_test = test_data(:,20);

    if(strcmp(algo_class,'gemm'))
        predict_flop_inst_count = flop_func_gemm(coeff_flop, test_data_copy);
    else
        if(strcmp(algo_class,'winograd'))
            predict_flop_inst_count = flop_func_winograd(coeff_flop, test_data_copy);
        end
    end
    test_data_copy(:,20) = predict_flop_inst_count;

    mspe_flop = sqrt(mean(((predict_flop_inst_count - Y_test)./Y_test) .^ 2));
    mse_flop = sqrt(mean(((predict_flop_inst_count - Y_test)) .^ 2));

    %% STEP4: use model for each type to predict the execution time of each kernel
    fprintf('Step4: Predicting Final Results...\n');
    test_data_copy(:,10) = 1;
    [row, col] = size(test_data);
    compute_test_data = []; real_time_compute = [];
    memory_test_data = [];  real_time_memory = [];
    under_test_data = [];    real_time_under = [];
    for i = 1:row   
        if(test_data_copy(i,19) == 0)       % compute-bound
            test_data_copy(i,10) = time_compute_bound(coeff_compute, test_data_copy(i,:));
            compute_test_data = [compute_test_data;test_data_copy(i,:)];
            real_time_compute = [real_time_compute;test_data(i,10)];
        else
            if(test_data_copy(i,19) == 1)   % memory-bound
                test_data_copy(i,10) = time_dram_bound(coeff_memory, test_data_copy(i,:));
                memory_test_data = [memory_test_data;test_data_copy(i,:)];
                real_time_memory = [real_time_memory;test_data(i,10)];
            else                        
                if(test_data_copy(i,19) == 2)   % under-utilized
                    test_data_copy(i,10) = time_resource_bound(coeff_under, test_data_copy(i,:));
                    under_test_data = [under_test_data;test_data_copy(i,:)];   
                    real_time_under = [real_time_under;test_data(i,10)];
                else
                    fprintf('==DEBUG== Unrecognized Kernel Type at row %d\n', i);
                end
            end
        end
    end

    [num_test_compute,~] = size(compute_test_data);
    [num_test_memory,~] = size(memory_test_data);
    [num_test_under,~] = size(under_test_data);

    if(num_test_compute ~= 0)
        mspe_compute = sqrt(mean(((compute_test_data(:,10) - real_time_compute)./real_time_compute) .^ 2));
        mse_compute = sqrt(mean(((compute_test_data(:,10) - real_time_compute)) .^ 2));
    else
        mspe_compute = 0;
        mse_compute = 0;
    end

    if(num_test_memory ~= 0)
        mspe_memory = sqrt(mean(((memory_test_data(:,10) - real_time_memory)./real_time_memory) .^ 2));
        mse_memory = sqrt(mean(((memory_test_data(:,10) - real_time_memory)) .^ 2));
    else
        mspe_memory = 0;
        mse_memory = 0;
    end

    if(num_test_under ~= 0)
        mspe_under = sqrt(mean(((under_test_data(:,10) - real_time_under)./real_time_under) .^ 2));
        mse_under = sqrt(mean(((under_test_data(:,10) - real_time_under)) .^ 2));
    else
        mspe_under = 0;
        mse_under = 0;
    end

    predict_time = test_data_copy(:,10);
    real_time = test_data(:,10);

    mspe_overall = sqrt(mean(((predict_time - real_time)./real_time) .^ 2));
    mse_overall = sqrt(mean(((predict_time - real_time)) .^ 2));

    % save predict_time & real_time
    csvwrite([predict_time_save_path, kernel_name,'.csv'], [real_time, predict_time]);
    fprintf('Finished! Predicted results of %s has been saved to %s%s.csv\n', kernel_name, predict_time_save_path, kernel_name);

    %% Print Results
    fprintf('============= Overall Results =============\n');
    fprintf('%s(%s)\t #train: %d\t #test: %d\n',kernel_name, device, train_set_size, test_set_size);
    fprintf('Compute-bound\t %.4f%%\t %.4f\n', mspe_compute*100, mse_compute);
    fprintf('Memory-bound\t %.4f%%\t %.4f\n', mspe_memory*100, mse_memory);
    fprintf('Under-utilized\t %.4f%%\t %.4f\n', mspe_under*100, mse_under);
    fprintf('Overall\t %.4f%%\t %.4f\n', mspe_overall*100, mse_overall);
end

%% algo4
kernel_names = ["algo4_16", "algo4_32", "algo4_64", "algo4_128", "algo4_256"];
for index = 1:5
    kernel_name = kernel_names{index};
    file_name = [dataset_path, kernel_name, suffix];
    [test_data,~]=xlsread(file_name,'test');
    conv_coeff = csvread([save_path, 'conv_', kernel_name ,'.csv']);
    real_time = test_data(:,10);
    test_data(:,10) = 1;
    predict_time = algo4_func(conv_coeff, test_data);
    
    csvwrite([predict_time_save_path, kernel_name,'.csv'], [real_time, predict_time]);
    fprintf('Finished! Predicted results of %s has been saved to %s%s.csv\n', kernel_name, predict_time_save_path, kernel_name);    
end

%% algo5
kernel_name = 'algo5';
file_name = [dataset_path, kernel_name, suffix];
[test_data,~]=xlsread(file_name,'test');
conv_coeff = csvread([save_path, 'conv_', kernel_name ,'.csv']);
real_time = test_data(:,10);
test_data(:,10) = 1;
predict_time = algo5_func(conv_coeff, test_data);

csvwrite([predict_time_save_path, kernel_name,'.csv'], [real_time, predict_time]);
fprintf('Finished! Predicted results of %s has been saved to %s%s.csv\n', kernel_name, predict_time_save_path, kernel_name);  


%% print overall results
total_real_time =[];
total_predict_time = [];

% algo0
result_1 = csvread([predict_time_save_path, 'algo0_32x32.csv']);
result_2 = csvread([predict_time_save_path, 'algo0_64x128.csv']);
result_3 = csvread([predict_time_save_path, 'algo0_128x128.csv']);
result = [result_1;result_2;result_3];

real_time = result(:,1);
predict_time = result(:,2);
mspe_overall = sqrt(mean(((predict_time - real_time)./real_time) .^ 2));
mse_overall = sqrt(mean(((predict_time - real_time)) .^ 2));
fprintf('[Algo0] Overall error : %.4f%%\t %.4f\n', mspe_overall*100, mse_overall);
total_real_time = [total_real_time;real_time];
total_predict_time = [total_predict_time;predict_time];

% algo1
result_1 = csvread([predict_time_save_path, 'algo1_128x32.csv']);
result_2 = csvread([predict_time_save_path, 'algo1_128x64.csv']);
result_3 = csvread([predict_time_save_path, 'algo1_128x128.csv']);
result = [result_1;result_2;result_3];

real_time = result(:,1);
predict_time = result(:,2);
mspe_overall = sqrt(mean(((predict_time - real_time)./real_time) .^ 2));
mse_overall = sqrt(mean(((predict_time - real_time)) .^ 2));
fprintf('[Algo1] Overall error : %.4f%%\t %.4f\n', mspe_overall*100, mse_overall);
total_real_time = [total_real_time;real_time];
total_predict_time = [total_predict_time;predict_time];

% algo2
result_1 = csvread([predict_time_save_path, 'algo2_32x32.csv']);
result_2 = csvread([predict_time_save_path, 'algo2_64x128.csv']);
result_3 = csvread([predict_time_save_path, 'algo2_128x128.csv']);
result = [result_1;result_2;result_3];

real_time = result(:,1);
predict_time = result(:,2);
mspe_overall = sqrt(mean(((predict_time - real_time)./real_time) .^ 2));
mse_overall = sqrt(mean(((predict_time - real_time)) .^ 2));
fprintf('[Algo2] Overall error : %.4f%%\t %.4f\n', mspe_overall*100, mse_overall);
total_real_time = [total_real_time;real_time];
total_predict_time = [total_predict_time;predict_time];

% algo4
result_1 = csvread([predict_time_save_path, 'algo4_16.csv']);
result_2 = csvread([predict_time_save_path, 'algo4_32.csv']);
result_3 = csvread([predict_time_save_path, 'algo4_64.csv']);
result_4 = csvread([predict_time_save_path, 'algo4_128.csv']);
result_5 = csvread([predict_time_save_path, 'algo4_256.csv']);
result = [result_1;result_2;result_3;result_4;result_5];

real_time = result(:,1);
predict_time = result(:,2);
mspe_overall = sqrt(mean(((predict_time - real_time)./real_time) .^ 2));
mse_overall = sqrt(mean(((predict_time - real_time)) .^ 2));
fprintf('[Algo4] Overall error : %.4f%%\t %.4f\n', mspe_overall*100, mse_overall);
total_real_time = [total_real_time;real_time];
total_predict_time = [total_predict_time;predict_time];

% algo5
result_1 = csvread([predict_time_save_path, 'algo5.csv']);
result = result_1;

real_time = result(:,1);
predict_time = result(:,2);
mspe_overall = sqrt(mean(((predict_time - real_time)./real_time) .^ 2));
mse_overall = sqrt(mean(((predict_time - real_time)) .^ 2));
fprintf('[Algo5] Overall error : %.4f%%\t %.4f\n', mspe_overall*100, mse_overall);
total_real_time = [total_real_time;real_time];
total_predict_time = [total_predict_time;predict_time];

% algo6
result_1 = csvread([predict_time_save_path, 'algo6_148.csv']);
result_2 = csvread([predict_time_save_path, 'algo6_228.csv']);
result_3 = csvread([predict_time_save_path, 'algo6_418.csv']);
result = [result_1;result_2;result_3];

real_time = result(:,1);
predict_time = result(:,2);
mspe_overall = sqrt(mean(((predict_time - real_time)./real_time) .^ 2));
mse_overall = sqrt(mean(((predict_time - real_time)) .^ 2));
fprintf('[Algo6] Overall error : %.4f%%\t %.4f\n', mspe_overall*100, mse_overall);
total_real_time = [total_real_time;real_time];
total_predict_time = [total_predict_time;predict_time];

% algo7
result_1 = csvread([predict_time_save_path, 'algo7_128x64_3x3.csv']);
result_2 = csvread([predict_time_save_path, 'algo7_128x64_5x5.csv']);
result_3 = csvread([predict_time_save_path, 'algo7_128x128_3x3.csv']);
result_4 = csvread([predict_time_save_path, 'algo7_128x128_5x5.csv']);
result = [result_1;result_2;result_3;result_4];

real_time = result(:,1);
predict_time = result(:,2);
mspe_overall = sqrt(mean(((predict_time - real_time)./real_time) .^ 2));
mse_overall = sqrt(mean(((predict_time - real_time)) .^ 2));
fprintf('[Algo7] Overall error : %.4f%%\t %.4f\n', mspe_overall*100, mse_overall);
total_real_time = [total_real_time;real_time];
total_predict_time = [total_predict_time;predict_time];

% total 
mspe_overall = sqrt(mean(((total_predict_time - total_real_time)./total_real_time) .^ 2));
mse_overall = sqrt(mean(((total_predict_time - total_real_time)) .^ 2));
fprintf('[total] Overall error : %.4f%%\t %.4f\n', mspe_overall*100, mse_overall);

