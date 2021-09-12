function [ pred_time ] = seer_predict_algo6( device, saved_model_path, test_data )

load([saved_model_path, 'ctree_algo6_148.mat'],'TypeClassificationTree');
ctree_algo6_148 = TypeClassificationTree;
load([saved_model_path, 'rtree_dread_algo6_148.mat'],'DramReadRegTree');
rtree_dread_algo6_148 = DramReadRegTree;
load([saved_model_path, 'rtree_dwrite_algo6_148.mat'],'DramWriteRegTree');
rtree_dwrite_algo6_148 = DramWriteRegTree;
flop_algo6_148 = csvread([saved_model_path, 'flop_algo6_148.csv']);
conv_algo6_148_compute = csvread([saved_model_path, 'conv_algo6_148_compute.csv']);
conv_algo6_148_memory = csvread([saved_model_path, 'conv_algo6_148_memory.csv']);
conv_algo6_148_under = csvread([saved_model_path, 'conv_algo6_148_under_utilized.csv']);

if(strcmp(device,'TitanXp'))
    load([saved_model_path, 'ctree_algo6_228.mat'],'TypeClassificationTree');
    ctree_algo6_228 = TypeClassificationTree;
    load([saved_model_path, 'rtree_dread_algo6_228.mat'],'DramReadRegTree');
    rtree_dread_algo6_228 = DramReadRegTree;
    load([saved_model_path, 'rtree_dwrite_algo6_228.mat'],'DramWriteRegTree');
    rtree_dwrite_algo6_228 = DramWriteRegTree;
    flop_algo6_228 = csvread([saved_model_path, 'flop_algo6_228.csv']);
    conv_algo6_228_compute = csvread([saved_model_path, 'conv_algo6_228_compute.csv'] );
    conv_algo6_228_memory = csvread([saved_model_path, 'conv_algo6_228_memory.csv'] );
    conv_algo6_228_under = csvread([saved_model_path, 'conv_algo6_228_under_utilized.csv'] );

    load([saved_model_path, 'ctree_algo6_418.mat'],'TypeClassificationTree');
    ctree_algo6_418 = TypeClassificationTree;
    load([saved_model_path, 'rtree_dread_algo6_418.mat'],'DramReadRegTree');
    rtree_dread_algo6_418 = DramReadRegTree;
    load([saved_model_path, 'rtree_dwrite_algo6_418.mat'],'DramWriteRegTree');
    rtree_dwrite_algo6_418 = DramWriteRegTree;
    flop_algo6_418 = csvread([saved_model_path, 'flop_algo6_418.csv'] );
    conv_algo6_418_compute = csvread([saved_model_path, 'conv_algo6_418_compute.csv'] );
    conv_algo6_418_memory = csvread([saved_model_path, 'conv_algo6_418_memory.csv'] );
    conv_algo6_418_under = csvread([saved_model_path, 'conv_algo6_418_under_utilized.csv']);

    load([saved_model_path, 'ctree_kernel_algo6.mat'],'Mdl');
    ctree_kernel_algo6 = Mdl;
end
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
% algo6_148
% calculate num_blocks
test_data(:,14) = 60;
test_data(:,15) = ceil(batch_size/1).* ceil(ceil(in_wid/2)/4).* ceil(ceil(in_wid/2)/8).* ceil(out_chan/32);
test_data(:,16) = ceil(test_data(:,15)./test_data(:,14));
% predict type
X_test = test_data(:, [2 3 4 5 6 7 15]);
[predict_type,~] = predict(ctree_algo6_148,X_test);
test_data(:,19) = predict_type;
% predict # of flop
test_data(:,20) = 1;
predict_flop_inst_count = flop_func_winograd(flop_algo6_148, test_data);
test_data(:,20) = predict_flop_inst_count;
% predict # of DRAM
X_test = test_data(:, [2 3 4 5 6 7]);
[predict_dram_read_ratio,~] = predict(rtree_dread_algo6_148,X_test);
test_data(:,23) = predict_dram_read_ratio;
test_data(:,21) = predict_dram_read_ratio .* in_chan;

[predict_dram_write_ratio,~] = predict(rtree_dwrite_algo6_148,X_test);
test_data(:,24) = predict_dram_write_ratio;
test_data(:,22) = test_data(:,24);

% predict overall time
test_data(:,10) = 1;
predict_time_1 = test_data(:,10)*200000;
predict_type_1 = test_data(:,19);
[row, ~] = size(test_data);
for i = 1:row 
    if(kernel_wid(i) == 3 && stride(i) == 1)
        if(test_data(i,19) == 0)       % compute-bound
            predict_time_1(i) = time_compute_bound(conv_algo6_148_compute, test_data(i,:));
        else
            if(test_data(i,19) == 1)   % memory-bound
                predict_time_1(i) = time_dram_bound(conv_algo6_148_memory, test_data(i,:));
            else                        
                if(test_data(i,19) == 2)   % under-bound
                    predict_time_1(i) = time_resource_bound(conv_algo6_148_under, test_data(i,:));
                else
                    fprintf('==ERROR== Unrecognized Kernel Type at row %d\n', i);
                end
            end
        end
    end
end

if(strcmp(device,'TitanXp'))
    % algo6_228
    % calculate num_blocks
    test_data(:,14) = 60;
    test_data(:,15) = ceil(batch_size/2).* ceil(ceil(in_wid/2)/2).* ceil(ceil(in_wid/2)/8).* ceil(out_chan/32);
    test_data(:,16) = ceil(test_data(:,15)./test_data(:,14));
    % predict type
    X_test = test_data(:, [2 3 4 5 6 7 15]);
    [predict_type,~] = predict(ctree_algo6_228,X_test);
    test_data(:,19) = predict_type;
    % predict # of flop
    test_data(:,20) = 1;
    predict_flop_inst_count = flop_func_winograd(flop_algo6_228, test_data);
    test_data(:,20) = predict_flop_inst_count;
    % predict # of DRAM
    X_test = test_data(:, [2 3 4 5 6 7]);
    [predict_dram_read_ratio,~] = predict(rtree_dread_algo6_228,X_test);
    test_data(:,23) = predict_dram_read_ratio;
    test_data(:,21) = predict_dram_read_ratio .* in_chan;

    [predict_dram_write_ratio,~] = predict(rtree_dwrite_algo6_228,X_test);
    test_data(:,24) = predict_dram_write_ratio;
    test_data(:,22) = test_data(:,24);

    % predict overall time
    test_data(:,10) = 1;
    predict_time_2 = test_data(:,10)*200000;
    predict_type_2 = test_data(:,19);
    [row, ~] = size(test_data);
    for i = 1:row   
        if(kernel_wid(i) == 3 && stride(i) == 1)
            if(test_data(i,19) == 0)       % compute-bound
                predict_time_2(i) = time_compute_bound(conv_algo6_228_compute, test_data(i,:));
            else
                if(test_data(i,19) == 1)   % memory-bound
                    predict_time_2(i) = time_dram_bound(conv_algo6_228_memory, test_data(i,:));
                else                        
                    if(test_data(i,19) == 2)   % under-bound
                        predict_time_2(i) = time_resource_bound(conv_algo6_228_under, test_data(i,:));
                    else
                        fprintf('==ERROR== Unrecognized Kernel Type at row %d\n', i);
                    end
                end
            end
        end
    end

    % algo6_418
    % calculate num_blocks
    test_data(:,14) = 60;
    test_data(:,15) = ceil(batch_size/4).* ceil(ceil(in_wid/2)/1).* ceil(ceil(in_wid/2)/8).* ceil(out_chan/32);
    test_data(:,16) = ceil(test_data(:,15)./test_data(:,14));
    % predict type
    X_test = test_data(:, [2 3 4 5 6 7 15]);
    [predict_type,~] = predict(ctree_algo6_418,X_test);
    test_data(:,19) = predict_type;
    % predict # of flop
    test_data(:,20) = 1;
    predict_flop_inst_count = flop_func_winograd(flop_algo6_418, test_data);
    test_data(:,20) = predict_flop_inst_count;
    % predict # of DRAM
    X_test = test_data(:, [2 3 4 5 6 7]);

    [predict_dram_read_ratio,~] = predict(rtree_dread_algo6_418,X_test);
    test_data(:,23) = predict_dram_read_ratio;
    test_data(:,21) = predict_dram_read_ratio .* in_chan;

    [predict_dram_write_ratio,~] = predict(rtree_dwrite_algo6_418,X_test);
    test_data(:,24) = predict_dram_write_ratio;
    test_data(:,22) = test_data(:,24);

    % predict overall time
    test_data(:,10) = 1;
    predict_time_3 = test_data(:,10)*200000;
    predict_type_3 = test_data(:,19);
    [row, ~] = size(test_data);
    for i = 1:row   
        if(kernel_wid(i) == 3 && stride(i) == 1)
            if(test_data(i,19) == 0)       % compute-bound
                predict_time_3(i) = time_compute_bound(conv_algo6_418_compute, test_data(i,:));
            else
                if(test_data(i,19) == 1)   % memory-bound
                    predict_time_3(i) = time_dram_bound(conv_algo6_418_memory, test_data(i,:));
                else                        
                    if(test_data(i,19) == 2)   % under-bound
                        predict_time_3(i) = time_resource_bound(conv_algo6_418_under, test_data(i,:));
                    else
                        fprintf('==ERROR== Unrecognized Kernel Type at row %d\n', i);
                    end
                end
            end
        end
    end
end
% predict kernel choice
predict_time_algo6 = test_data(:,10)*10000;
predict_type_algo6 = ones(row,1)*(-1);
if(strcmp(device,'TitanXp'))
    [predict_kernel_type,~] = predict(ctree_kernel_algo6, test_data(:,[2 3 4 5 6 7]));
end
if(strcmp(device,'TitanV'))
    predict_kernel_type = zeros(row,1);
end
for i = 1: row
    if(predict_kernel_type(i) == 0)
        predict_time_algo6(i) = predict_time_1(i);
        predict_type_algo6(i) = predict_type_1(i);
    elseif(predict_kernel_type(i) == 1)
        predict_time_algo6(i) = predict_time_2(i);
        predict_type_algo6(i) = predict_type_2(i);
    else
        predict_time_algo6(i) = predict_time_3(i);  
        predict_type_algo6(i) = predict_type_3(i);
    end 
end

test_data(:,10) = predict_time_algo6;
test_data(:,11) = predict_type_algo6;

% csvwrite(['../../final-evaluation-data/',device,'_predict_type_algo6.csv'], test_data);
pred_time = test_data;

% fprintf("Algo6 prediction finished.\n");

end