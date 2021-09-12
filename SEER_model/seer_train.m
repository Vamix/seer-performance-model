
%% read source data
% file format: 
%   1      2      3       4       5       6      7     8   9    10  
% batch in_chan in_wid out_chan out_wid kernel stride pad algo time 
%   11		   12    	  13   	 	  14     	    15        16      17      18     
% inst_fp  dram_read  dram_write  #block-max  #block_launch  #iter  fp_uti  dram_uti  

% to be computed in step0: 
%  19        20            21             22                23              24
% type  inst_fp_blk  dram_read_blk  dram_write_blk   dram_read_ratio  dram_write_ratio
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

%% training 
for index = 1: num_kernels
    kernel_name = kernel_names{index};
    fprintf('======= kernel: %s =======\n', kernel_name);    
    file_name = [dataset_path, kernel_name, suffix];
    [train_data,~]=xlsread(file_name,1);
    %% STEP0: Data preprocess
    [train_set_size,~] = size(train_data);

    if(strncmp(kernel_name,'algo0',5) || strncmp(kernel_name,'algo1',5) || strncmp(kernel_name,'algo2',5))
        algo_class = 'gemm'; 
    else
        if(strncmp(kernel_name,'algo6',5) || strncmp(kernel_name,'algo7',5))
            algo_class = 'winograd'; 
        end
    end

    % A trick to improve dram regression tree accuracy.
    if(strncmp(kernel_name,'algo1_128x32',12))
        for i = 1:train_set_size
            if(train_data(i,4) < 32)
                train_data(i,4) = 32;
            end
        end
    end
    
    train_data(:,20) = train_data(:,11)./train_data(:,15);
    train_data(:,21) = train_data(:,12)./train_data(:,15);
    train_data(:,22) = train_data(:,13)./train_data(:,15);
    if(strcmp(algo_class,'gemm'))
        train_data(:,23) = train_data(:,21)./(train_data(:,6).^2 .* train_data(:,2));
        train_data(:,24) = train_data(:,22);
    else
        if(strcmp(algo_class,'winograd'))
            train_data(:,23) = train_data(:,21)./train_data(:,2);
            train_data(:,24) = train_data(:,22);        
        end
    end

    [row, ~] = size(train_data);
    for i = 1:row   
        if(train_data(i,17) >= 8)       % compute-bound
            train_data(i,19) = 0;
        else
            if(train_data(i,18) >= 8)   % memory-bound
                train_data(i,19) = 1;
            else                        % under-utilized
                train_data(i,19) = 2;
            end
        end
    end
    train_data(:,16) = ceil(train_data(:,15)./train_data(:,14));

    %% STEP1: train classification tree
    fprintf('Step1: Training classification tree....\n');
    X_train = train_data(:, [2 3 4 5 6 7 15]);
    Y_train = train_data(:, 19);
    TypeClassificationTree = fitctree(X_train,Y_train, 'PredictorNames',{'in_chan','in_wid','out_chan','out_wid','kernel', 'stride', 'num_block'},'OptimizeHyperparameters','auto','OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',struct('MaxObjectiveEvaluations',30,'ShowPlots',false, 'Verbose',0), 'MaxNumSplits', 64);
    tree_name = [save_path,'ctree_',kernel_name,'.mat'];
    save(tree_name,'TypeClassificationTree');

    %% STEP2: train regression tree for dram_read & dram_write
    fprintf('Step2: Training regression tree for #dram transactions....\n');
    if(strcmp(algo_class,'gemm'))
        X_train = train_data(:, [2 3 4 5 6 7 15]);
        size_one_kernel = train_data(:,6).* train_data(:,6).* train_data(:,2);
        size_all_kernel = size_one_kernel.*train_data(:,4);
        mid_size = train_data(:,5).* train_data(:,5).* train_data(:,2).*train_data(:,6).* train_data(:,6);
        X_train = [X_train,size_one_kernel,size_all_kernel,mid_size];

        Y_train = train_data(:, 23);
        DramReadRegTree = fitrtree(X_train,Y_train,'PredictorNames',{'in_chan','in_wid','out_chan','out_wid','kernel', 'stride', 'num_block','size_one_kernel', 'size_all_kernel', 'mid_size'},'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',struct('MaxObjectiveEvaluations',30,'ShowPlots',false, 'Verbose',0), 'MaxNumSplits', 64);
        Y_train = train_data(:, 24);
        DramWriteRegTree = fitrtree(X_train,Y_train,'PredictorNames',{'in_chan','in_wid','out_chan','out_wid','kernel', 'stride', 'num_block','size_one_kernel', 'size_all_kernel', 'mid_size'},'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',struct('MaxObjectiveEvaluations',30,'ShowPlots',false, 'Verbose',0), 'MaxNumSplits', 64);
    else
        if(strcmp(algo_class,'winograd'))
            X_train = train_data(:, [2 3 4 5 6 7]);
            Y_train = train_data(:, 23);
            DramReadRegTree = fitrtree(X_train,Y_train,'PredictorNames',{'in_chan','in_wid','out_chan','out_wid','kernel', 'stride'},'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',struct('MaxObjectiveEvaluations',30,'ShowPlots',false, 'Verbose',0), 'MaxNumSplits', 64);
            Y_train = train_data(:, 24);
            DramWriteRegTree = fitrtree(X_train,Y_train,'PredictorNames',{'in_chan','in_wid','out_chan','out_wid','kernel', 'stride'},'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',struct('MaxObjectiveEvaluations',30,'ShowPlots',false, 'Verbose',0), 'MaxNumSplits', 64);
        end
    end

    tree_name = [save_path, 'rtree_dread_',kernel_name,'.mat'];
    save(tree_name,'DramReadRegTree');
    tree_name = [save_path, 'rtree_dwrite_',kernel_name,'.mat'];
    save(tree_name,'DramWriteRegTree');

    %% STEP3: train poly model for #inst_flop_sp
    fprintf('Step3: Fitting #flop_inst Coeffs....\n');
    coeff0 = ones(1,2);
    [row, ~] = size(train_data);
    y_train = ones(row,1);

    if(strcmp(algo_class,'gemm'))
        [coeff_flop, ~, ~] = nlinfit(train_data, y_train, @flop_func_gemm, coeff0);
    else
        if(strcmp(algo_class,'winograd'))
            [coeff_flop, ~, ~] = nlinfit(train_data, y_train, @flop_func_winograd, coeff0);
        end
    end
    csvwrite([save_path, 'flop_',kernel_name,'.csv'], coeff_flop);

    %% STEP4: fitting model coeffs for 3 types of kernels: compute/memory/under-utilized
    fprintf('Step4: Fitting Model Coeffs....\n');
    [row, ~] = size(train_data);
    compute_train_data = [];
    memory_train_data = [];
    under_train_data = [];
    for i = 1:row   
        if(train_data(i,19) == 0)       % compute-bound
            compute_train_data = [compute_train_data;train_data(i,:)];
        else
            if(train_data(i,19) == 1)   % memory-bound
                memory_train_data = [memory_train_data;train_data(i,:)];
            else                        % under-utilized
                under_train_data = [under_train_data;train_data(i,:)];
            end
        end
    end

    % train compute-bound models:
    fprintf('Compute-bound Model fitting...\n');
    coeff0 = ones(1,2);
    [num_train_compute, ~] = size(compute_train_data);
    if(num_train_compute ~= 0)
        y_train = ones(num_train_compute,1);
        [coeff_compute, ~, ~] = nlinfit(compute_train_data, y_train, @time_compute_bound, coeff0);
        csvwrite([save_path, 'conv_',kernel_name,'_compute.csv'], coeff_compute);
    else
        fprintf('==WARNING== There is no compute-bound sample in training set.\n');
        coeff_compute = coeff0;
        csvwrite([save_path, 'conv_',kernel_name,'_compute.csv'], coeff_compute);
    end

    % train memory-bound models:
    fprintf('Memory-bound Model fitting...\n');
    coeff0 = ones(1,2);
    [num_train_memory, ~] = size(memory_train_data);
    if(num_train_memory ~= 0)
        y_train = ones(num_train_memory,1);
        [coeff_memory, ~, ~] = nlinfit(memory_train_data, y_train, @time_dram_bound, coeff0);
        csvwrite([save_path, 'conv_',kernel_name,'_memory.csv'], coeff_memory);
    else
        fprintf('==WARNING== There is no memory-bound sample in training set.\n');
        coeff_memory = coeff0;
        csvwrite([save_path, 'conv_',kernel_name,'_memory.csv'], coeff_memory);
    end    

    % train under-utilized models:
    fprintf('Under-utilized Model fitting...\n');
    coeff0 = ones(1,4);
    [num_train_under, ~] = size(under_train_data);
    if(num_train_under ~= 0)
        y_train = ones(num_train_under,1);
        [coeff_under, ~, ~] = nlinfit(under_train_data, y_train, @time_resource_bound, coeff0);
        csvwrite([save_path, 'conv_',kernel_name,'_under_utilized.csv'], coeff_under);
    else
        fprintf('==WARNING== There is no under-utilized sample in training set.\n');
        coeff_under = coeff0;
        csvwrite([save_path, 'conv_',kernel_name,'_under_utilized.csv'], coeff_under);
    end    

    fprintf('Model params fitting for %s (on device %s) is finished!!!\n', kernel_name, device );

end

%% poly model for algo4
kernel_names = ["algo4_16", "algo4_32", "algo4_64", "algo4_128", "algo4_256"];
for index = 1:5 
    kernel_name = kernel_names{index};
    fprintf('======= kernel: %s =======\n', kernel_name);  
    file_name = [dataset_path, kernel_name, suffix];
    [train_data,~]=xlsread(file_name,1);
    coeff0 = ones(1,4);
    [row, ~] = size(train_data);
    y_train = ones(row,1);
    [coeff_conv, ~, ~] = nlinfit(train_data, y_train, @algo4_func, coeff0);
    csvwrite([save_path, 'conv_', kernel_name ,'.csv'], coeff_conv);  
    fprintf('Model params fitting for %s kernels (on %s) is finished!!!\n', kernel_name, device); 
end

%% poly model for algo5
kernel_name = 'algo5';
fprintf('======= kernel: %s =======\n', kernel_name);   
file_name = [dataset_path, kernel_name, suffix];
[train_data,~]=xlsread(file_name,1);
coeff0 = ones(1,4);
[row, ~] = size(train_data);
y_train = ones(row,1);
[coeff_conv, ~, ~] = nlinfit(train_data, y_train, @algo5_func, coeff0);
csvwrite([save_path, 'conv_', kernel_name ,'.csv'], coeff_conv);   
fprintf('Model params fitting for %s kernels (on %s) is finished!!!\n', kernel_name, device); 


%% poly model for algo2-pre kernels  
kernel_name = 'algo2_pre';
fprintf('======= kernel: %s =======\n', kernel_name); 
file_name = [dataset_path, kernel_name, suffix];
[train_data_pre,~]=xlsread(file_name,1);
coeff0 = ones(1,3);
[row, ~] = size(train_data_pre);
y_train = ones(row,1);
[coeff_pre, ~, ~] = nlinfit(train_data_pre, y_train, @pre_kernel_algo2, coeff0);
csvwrite([save_path,'conv_algo2_pre.csv'], coeff_pre);    
fprintf('Model fitting for algo2-pre kernels (on %s) is finished!!!\n', device );

%% poly model for algo7-pre kernel
kernel_names = ["algo7_128x64_3x3", "algo7_128x64_5x5", "algo7_128x128_3x3", "algo7_128x128_5x5"];

for index = 1:4
    kernel_name = kernel_names{index};
    fprintf('======= kernel: %s_pre =======\n', kernel_name); 
    file_name = [dataset_path, kernel_name, '_pre', suffix];
    [train_data_pre,~]=xlsread(file_name,1);
    coeff0 = ones(1,4);
    [row, ~] = size(train_data_pre);
    y_train = ones(row,1);
    [coeff_pre, ~, ~] = nlinfit(train_data_pre, y_train, @pre_kernel_algo7, coeff0);
    csvwrite([save_path,'conv_', kernel_name ,'_pre.csv'], coeff_pre);   
    fprintf('Model fitting for %s-pre kernels (on %s) is finished!!!\n', kernel_name, device);    
end

%% kernel-ctree for algo0 (used for choosing kernel with different tile size)
fprintf('======= kernel-choosing tree for algo0 =======\n'); 
if(strcmp(device, 'TitanXp'))
    kernel_name = 'algo0_32x32';
    file_name = [dataset_path,kernel_name,suffix];
    [train_data_0,~]=xlsread(file_name,1);
    train_data_0(:, 20) = 0;
    kernel_name = 'algo0_64x128';
    file_name = [dataset_path,kernel_name,suffix];
    [train_data_1,~]=xlsread(file_name,1);
    train_data_1(:, 20) = 1;
    kernel_name = 'algo0_128x128';
    file_name = [dataset_path,kernel_name,suffix];
    [train_data_2,~]=xlsread(file_name,1);
    train_data_2(:, 20) = 2;
    train_data = [train_data_0;train_data_1;train_data_2];
else
    if(strcmp(device, 'TitanV'))
        kernel_name = 'algo0_32x32';
        file_name = [dataset_path,kernel_name,suffix];
        [train_data_0,~]=xlsread(file_name,1);
        train_data_0(:, 20) = 0;
        kernel_name = 'algo0_64x128';
        file_name = [dataset_path,kernel_name,suffix];
        [train_data_1,~]=xlsread(file_name,1);
        train_data_1(:, 20) = 1;
        train_data = [train_data_0;train_data_1];        
    end
end

X_train = train_data(:,[2 3 4 5 6 7]);
Y_train = train_data(:,20);

Mdl = fitctree(X_train,Y_train, 'PredictorNames',{'in_chan','in_wid','out_chan','out_wid','kernel', 'stride'},'OptimizeHyperparameters','auto','OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',struct('MaxObjectiveEvaluations',30, 'ShowPlots',false, 'Verbose',0), 'MaxNumSplits', 64);
save([save_path,'ctree_kernel_algo0.mat'],'Mdl');
fprintf('Kernel-choosing tree for algo0 kernels (on %s) is finished!!!\n', device); 

%% kernel-ctree for algo2
fprintf('======= kernel-choosing tree for algo2 =======\n'); 
if(strcmp(device, 'TitanXp'))
    kernel_name = 'algo2_32x32';
    file_name = [dataset_path,kernel_name,suffix];
    [train_data_0,~]=xlsread(file_name,1);
    train_data_0(:, 20) = 0;
    kernel_name = 'algo2_64x128';
    file_name = [dataset_path,kernel_name,suffix];
    [train_data_1,~]=xlsread(file_name,1);
    train_data_1(:, 20) = 1;
    kernel_name = 'algo2_128x128';
    file_name = [dataset_path,kernel_name,suffix];
    [train_data_2,~]=xlsread(file_name,1);
    train_data_2(:, 20) = 2;
    train_data = [train_data_0;train_data_1;train_data_2];
else
    if(strcmp(device, 'TitanV'))
        kernel_name = 'algo2_32x32';
        file_name = [dataset_path,kernel_name,suffix];
        [train_data_0,~]=xlsread(file_name,1);
        train_data_0(:, 20) = 0;
        kernel_name = 'algo2_64x128';
        file_name = [dataset_path,kernel_name,suffix];
        [train_data_1,~]=xlsread(file_name,1);
        train_data_1(:, 20) = 1;
        train_data = [train_data_0;train_data_1]; 
    end
end

X_train = train_data(:,[2 3 4 5 6 7]);
Y_train = train_data(:,20);

Mdl = fitctree(X_train,Y_train, 'PredictorNames',{'in_chan','in_wid','out_chan','out_wid','kernel', 'stride'},'OptimizeHyperparameters','auto','OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',struct('MaxObjectiveEvaluations',30, 'ShowPlots',false, 'Verbose',0), 'MaxNumSplits', 64);
save([save_path,'ctree_kernel_algo2.mat'],'Mdl');
fprintf('Kernel-choosing tree for algo2 (on %s) is finished!!!\n', device); 

%% kernel-ctree for algo6
fprintf('======= kernel-choosing tree for algo6 =======\n'); 
kernel_name = 'algo6_148';
file_name = [dataset_path,kernel_name,suffix];
[train_data_0,~]=xlsread(file_name,1);
train_data_0(:, 20) = 0;
kernel_name = 'algo6_228';
file_name = [dataset_path,kernel_name,suffix];
[train_data_1,~]=xlsread(file_name,1);
train_data_1(:, 20) = 1;
kernel_name = 'algo6_418';
file_name = [dataset_path,kernel_name,suffix];
[train_data_2,~]=xlsread(file_name,1);
train_data_2(:, 20) = 2;
train_data = [train_data_0;train_data_1;train_data_2];
% train_data = [train_data_0;train_data_2];

X_train = train_data(:,[2 3 4 5 6 7]);
Y_train = train_data(:,20);

Mdl = fitctree(X_train,Y_train, 'PredictorNames',{'in_chan','in_wid','out_chan','out_wid','kernel', 'stride'},'OptimizeHyperparameters','auto','OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',struct('MaxObjectiveEvaluations',30, 'ShowPlots',false, 'Verbose',0), 'MaxNumSplits', 64);
save([save_path,'ctree_kernel_algo6.mat'],'Mdl');
fprintf('Kernel-choosing tree for algo6 (on %s) is finished!!!\n', device);  

%% kernel-ctree for algo7
fprintf('======= kernel-choosing tree for algo7 =======\n'); 
kernel_name = 'algo7_128x64_3x3';
file_name = [dataset_path,kernel_name,suffix];
[train_data_0,~]=xlsread(file_name,1);
train_data_0(:, 20) = 0;
kernel_name = 'algo7_128x64_5x5';
file_name = [dataset_path,kernel_name,suffix];
[train_data_1,~]=xlsread(file_name,1);
train_data_1(:, 20) = 0;
kernel_name = 'algo7_128x128_3x3';
file_name = [dataset_path,kernel_name,suffix];
[train_data_2,~]=xlsread(file_name,1);
train_data_2(:, 20) = 1;
kernel_name = 'algo7_128x128_5x5';
file_name = [dataset_path,kernel_name,suffix];
[train_data_3,~]=xlsread(file_name,1);
train_data_3(:, 20) = 1;
train_data = [train_data_0;train_data_1;train_data_2;train_data_3];

X_train = train_data(:,[2 3 4 5 6 7]);
Y_train = train_data(:,20);

Mdl = fitctree(X_train,Y_train, 'PredictorNames',{'in_chan','in_wid','out_chan','out_wid','kernel', 'stride'},'OptimizeHyperparameters','auto','OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',struct('MaxObjectiveEvaluations',30, 'ShowPlots',false, 'Verbose',0), 'MaxNumSplits', 64);
save([save_path, 'ctree_kernel_algo7.mat'],'Mdl');
fprintf('Kernel-choosing tree for algo7 (on %s) is finished!!!\n', device); 

%% kernel type tree for algo5
fprintf('======= kernel-classification tree for algo5 =======\n'); 
file_name = [dataset_path,'algo5-ctree', suffix];
[train_data,~]=xlsread(file_name,1);
X_train = train_data(:, [2 4 5 6]);
% Y_train = train_data(:, 19);
Y_train = train_data(:, 10);
TypeClassificationTree = fitctree(X_train,Y_train, 'PredictorNames',{'in_chan','out_chan','out_wid','kernel'},'OptimizeHyperparameters','auto','OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',struct('MaxObjectiveEvaluations',30, 'ShowPlots',false, 'Verbose',0), 'MaxNumSplits', 64);
tree_name = [save_path,'ctree_algo5.mat'];
save(tree_name,'TypeClassificationTree');
fprintf('Classification tree for algo5 (on %s) is finished!!!\n', device); 

