function [ pred_time ] = seer_predict_algo2( device, saved_model_path, test_data )

load([saved_model_path, 'ctree_algo2_32x32.mat'],'TypeClassificationTree');
ctree_algo2_32x32 = TypeClassificationTree;
load([saved_model_path, 'rtree_dread_algo2_32x32.mat'],'DramReadRegTree');
rtree_dread_algo2_32x32 = DramReadRegTree;
load([saved_model_path, 'rtree_dwrite_algo2_32x32.mat'],'DramWriteRegTree');
rtree_dwrite_algo2_32x32 = DramWriteRegTree;
flop_algo2_32x32 = csvread([saved_model_path, 'flop_algo2_32x32.csv']);
conv_algo2_32x32_compute = csvread([saved_model_path, 'conv_algo2_32x32_compute.csv']);
conv_algo2_32x32_memory = csvread([saved_model_path, 'conv_algo2_32x32_memory.csv']);
conv_algo2_32x32_under = csvread([saved_model_path, 'conv_algo2_32x32_under_utilized.csv']);

load([saved_model_path, 'ctree_algo2_64x128.mat'],'TypeClassificationTree');
ctree_algo2_64x128 = TypeClassificationTree;
load([saved_model_path, 'rtree_dread_algo2_64x128.mat'],'DramReadRegTree');
rtree_dread_algo2_64x128 = DramReadRegTree;
load([saved_model_path, 'rtree_dwrite_algo2_64x128.mat'],'DramWriteRegTree');
rtree_dwrite_algo2_64x128 = DramWriteRegTree;
flop_algo2_64x128 = csvread([saved_model_path, 'flop_algo2_64x128.csv']);
conv_algo2_64x128_compute = csvread([saved_model_path, 'conv_algo2_64x128_compute.csv'] );
conv_algo2_64x128_memory = csvread([saved_model_path, 'conv_algo2_64x128_memory.csv'] );
conv_algo2_64x128_under = csvread([saved_model_path, 'conv_algo2_64x128_under_utilized.csv'] );

if(strcmp(device,'TitanXp'))
    load([saved_model_path, 'ctree_algo2_128x128.mat'],'TypeClassificationTree');
    ctree_algo2_128x128 = TypeClassificationTree;
    load([saved_model_path, 'rtree_dread_algo2_128x128.mat'],'DramReadRegTree');
    rtree_dread_algo2_128x128 = DramReadRegTree;
    load([saved_model_path, 'rtree_dwrite_algo2_128x128.mat'],'DramWriteRegTree');
    rtree_dwrite_algo2_128x128 = DramWriteRegTree;
    flop_algo2_128x128 = csvread([saved_model_path, 'flop_algo2_128x128.csv'] );
    conv_algo2_128x128_compute = csvread([saved_model_path, 'conv_algo2_128x128_compute.csv']);
    conv_algo2_128x128_memory = csvread([saved_model_path, 'conv_algo2_128x128_memory.csv'] );
    conv_algo2_128x128_under = csvread([saved_model_path, 'conv_algo2_128x128_under_utilized.csv']);

    coeff_pre = csvread([saved_model_path, 'conv_algo2_pre.csv']);
end

load([saved_model_path, 'ctree_kernel_algo2.mat'],'Mdl');
ctree_kernel_algo2 = Mdl;

%% read test data
% [test_data,~]=xlsread(file_name,1);

batch_size = test_data(:,1);
in_chan = test_data(:,2);
in_wid = test_data(:,3);
out_chan = test_data(:,4);
out_wid = test_data(:,5);
kernel_wid = test_data(:,6);
stride = test_data(:,7);

%% use coressponding model to predict for each algo, each variant.

% algo2_32x32
% calculate num_blocks
test_data(:,14) = 480;
test_data(:,15) = ceil((batch_size.*(out_wid.^2))/32).* ceil(out_chan/32);
test_data(:,16) = ceil(test_data(:,15)./test_data(:,14));
% predict type
X_test = test_data(:, [2 3 4 5 6 7 15]);
[predict_type,~] = predict(ctree_algo2_32x32,X_test);
test_data(:,19) = predict_type;
% predict # of flop
test_data(:,20) = 1;
predict_flop_inst_count = flop_func_gemm(flop_algo2_32x32, test_data);
test_data(:,20) = predict_flop_inst_count;
% predict # of DRAM
X_test = test_data(:, [2 3 4 5 6 7 15]);
size_one_kernel = kernel_wid.^2.* in_chan;
size_all_kernel = size_one_kernel .* out_chan;
mid_size = out_wid.^2 .* kernel_wid.^2 .* in_chan;
X_test = [X_test,size_one_kernel,size_all_kernel,mid_size];

[predict_dram_read_ratio,~] = predict(rtree_dread_algo2_32x32,X_test);
test_data(:,23) = predict_dram_read_ratio;
test_data(:,21) = predict_dram_read_ratio .* kernel_wid.^2 .* in_chan;

[predict_dram_write_ratio,~] = predict(rtree_dwrite_algo2_32x32,X_test);
test_data(:,24) = predict_dram_write_ratio;
test_data(:,22) = test_data(:,24);

% predict overall time
test_data(:,10) = 1;
predict_time_1 = test_data(:,10);
predict_type_1 = test_data(:,19);
[row, ~] = size(test_data);
for i = 1:row   
    if(test_data(i,19) == 0)       % compute-bound
        predict_time_1(i) = time_compute_bound(conv_algo2_32x32_compute, test_data(i,:));
    else
        if(test_data(i,19) == 1)   % memory-bound
            predict_time_1(i) = time_dram_bound(conv_algo2_32x32_memory, test_data(i,:));
        else                        
            if(test_data(i,19) == 2)   % under-bound
                predict_time_1(i) = time_resource_bound(conv_algo2_32x32_under, test_data(i,:));
            else
                fprintf('==ERROR== Unrecognized Kernel Type at row %d\n', i);
            end
        end
    end
end

% algo2_64x128
% calculate num_blocks
test_data(:,14) = 60;
test_data(:,15) = ceil((batch_size.*(out_wid.^2))/64).* ceil(out_chan/128);
test_data(:,16) = ceil(test_data(:,15)./test_data(:,14));
% predict type
X_test = test_data(:, [2 3 4 5 6 7 15]);
[predict_type,~] = predict(ctree_algo2_64x128,X_test);
test_data(:,19) = predict_type;
% predict # of flop
test_data(:,20) = 1;
predict_flop_inst_count = flop_func_gemm(flop_algo2_64x128, test_data);
test_data(:,20) = predict_flop_inst_count;
% predict # of DRAM
X_test = test_data(:, [2 3 4 5 6 7 15]);
size_one_kernel = kernel_wid.^2.* in_chan;
size_all_kernel = size_one_kernel .* out_chan;
mid_size = out_wid.^2 .* kernel_wid.^2 .* in_chan;
X_test = [X_test,size_one_kernel,size_all_kernel,mid_size];

[predict_dram_read_ratio,~] = predict(rtree_dread_algo2_64x128,X_test);
test_data(:,23) = predict_dram_read_ratio;
test_data(:,21) = predict_dram_read_ratio .* kernel_wid.^2 .* in_chan;

[predict_dram_write_ratio,~] = predict(rtree_dwrite_algo2_64x128,X_test);
test_data(:,24) = predict_dram_write_ratio;
test_data(:,22) = test_data(:,24);

% predict overall time
test_data(:,10) = 1;
predict_time_2 = test_data(:,10);
predict_type_2 = test_data(:,19);
[row, ~] = size(test_data);
for i = 1:row   
    if(test_data(i,19) == 0)       % compute-bound
        predict_time_2(i) = time_compute_bound(conv_algo2_64x128_compute, test_data(i,:));
    else
        if(test_data(i,19) == 1)   % memory-bound
            predict_time_2(i) = time_dram_bound(conv_algo2_64x128_memory, test_data(i,:));
        else                        
            if(test_data(i,19) == 2)   % under-bound
                predict_time_2(i) = time_resource_bound(conv_algo2_64x128_under, test_data(i,:));
            else
                fprintf('==ERROR== Unrecognized Kernel Type at row %d\n', i);
            end
        end
    end
end

% algo2_128x128
if(strcmp(device,'TitanXp'))
    % calculate num_blocks
    test_data(:,14) = 60;
    test_data(:,15) = ceil((batch_size.*(out_wid.^2))/128).* ceil(out_chan/128);
    test_data(:,16) = ceil(test_data(:,15)./test_data(:,14));
    % predict type
    X_test = test_data(:, [2 3 4 5 6 7 15]);
    [predict_type,~] = predict(ctree_algo2_128x128,X_test);
    test_data(:,19) = predict_type;
    % predict # of flop
    test_data(:,20) = 1;
    predict_flop_inst_count = flop_func_gemm(flop_algo2_128x128, test_data);
    test_data(:,20) = predict_flop_inst_count;
    % predict # of DRAM
    X_test = test_data(:, [2 3 4 5 6 7 15]);
    size_one_kernel = kernel_wid.^2.* in_chan;
    size_all_kernel = size_one_kernel .* out_chan;
    mid_size = out_wid.^2 .* kernel_wid.^2 .* in_chan;
    X_test = [X_test,size_one_kernel,size_all_kernel,mid_size];

    [predict_dram_read_ratio,~] = predict(rtree_dread_algo2_128x128,X_test);
    test_data(:,23) = predict_dram_read_ratio;
    test_data(:,21) = predict_dram_read_ratio .* kernel_wid.^2 .* in_chan;

    [predict_dram_write_ratio,~] = predict(rtree_dwrite_algo2_128x128,X_test);
    test_data(:,24) = predict_dram_write_ratio;
    test_data(:,22) = test_data(:,24);

    % predict overall time
    test_data(:,10) = 1;
    predict_time_3 = test_data(:,10);
    predict_type_3 = test_data(:,19);
    [row, ~] = size(test_data);
    for i = 1:row   
        if(test_data(i,19) == 0)       % compute-bound
            predict_time_3(i) = time_compute_bound(conv_algo2_128x128_compute, test_data(i,:));
        else
            if(test_data(i,19) == 1)   % memory-bound
                predict_time_3(i) = time_dram_bound(conv_algo2_128x128_memory, test_data(i,:));
            else                        
                if(test_data(i,19) == 2)   % under-bound
                    predict_time_3(i) = time_resource_bound(conv_algo2_128x128_under, test_data(i,:));
                else
                    fprintf('==ERROR== Unrecognized Kernel Type at row %d\n', i);
                end
            end
        end
    end
end

% predict pre-kernel time
% test_data_pre_copy = test_data_pre;
test_data(:,10) = 1;

if(strcmp(device,'TitanXp'))
    predict_pre_kernel_time = pre_kernel_algo2(coeff_pre, test_data);
    predict_time_1 = predict_time_1 + predict_pre_kernel_time;
    predict_time_2 = predict_time_2 + predict_pre_kernel_time;
    predict_time_3 = predict_time_3 + predict_pre_kernel_time;
end
% predict kernel choice
predict_time_algo2 = test_data(:,10)*10000;
predict_type_algo2 = [];

[predict_kernel_type,~] = predict(ctree_kernel_algo2, test_data(:,[2 3 4 5 6 7]));
for i = 1: row
    if(predict_kernel_type(i) == 0)
        predict_time_algo2(i) = predict_time_1(i);
        predict_type_algo2(i) = predict_type_1(i);
    elseif(predict_kernel_type(i) == 1)
        predict_time_algo2(i) = predict_time_2(i);
        predict_type_algo2(i) = predict_type_2(i);
    else
        if(strcmp(device,'TitanXp'))
            predict_time_algo2(i) = predict_time_3(i);    
            predict_type_algo2(i) = predict_type_3(i);
        end
    end 
end

test_data(:,10) = predict_time_algo2;
test_data(:,11) = predict_type_algo2;

% csvwrite(['../../final-evaluation-data/',device,'_predict_type_algo2.csv'], test_data);
pred_time = test_data;

% fprintf("Algo2 prediction finished.\n");

end