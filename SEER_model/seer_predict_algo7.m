function [ pred_time ] = seer_predict_algo7( device, saved_model_path, test_data )

load([saved_model_path, 'ctree_algo7_128x64_3x3.mat'],'TypeClassificationTree');
ctree_algo7_128x64_3x3 = TypeClassificationTree;
load([saved_model_path, 'rtree_dread_algo7_128x64_3x3.mat'],'DramReadRegTree');
rtree_dread_algo7_128x64_3x3 = DramReadRegTree;
load([saved_model_path, 'rtree_dwrite_algo7_128x64_3x3.mat'],'DramWriteRegTree');
rtree_dwrite_algo7_128x64_3x3 = DramWriteRegTree;
flop_algo7_128x64_3x3 = csvread([saved_model_path, 'flop_algo7_128x64_3x3.csv']);
conv_algo7_128x64_3x3_compute = csvread([saved_model_path, 'conv_algo7_128x64_3x3_compute.csv']);
conv_algo7_128x64_3x3_memory = csvread([saved_model_path, 'conv_algo7_128x64_3x3_memory.csv']);
conv_algo7_128x64_3x3_under = csvread([saved_model_path, 'conv_algo7_128x64_3x3_under_utilized.csv']);

if(strcmp(device,'TitanXp'))
    coeff_pre_algo7_128x64_3x3 = csvread([saved_model_path, 'conv_algo7_128x64_3x3_pre.csv']);
end
load([saved_model_path, 'ctree_algo7_128x64_5x5.mat'],'TypeClassificationTree');
ctree_algo7_128x64_5x5 = TypeClassificationTree;
load([saved_model_path, 'rtree_dread_algo7_128x64_5x5.mat'],'DramReadRegTree');
rtree_dread_algo7_128x64_5x5 = DramReadRegTree;
load([saved_model_path, 'rtree_dwrite_algo7_128x64_5x5.mat'],'DramWriteRegTree');
rtree_dwrite_algo7_128x64_5x5 = DramWriteRegTree;
flop_algo7_128x64_5x5 = csvread([saved_model_path, 'flop_algo7_128x64_5x5.csv']);
conv_algo7_128x64_5x5_compute = csvread([saved_model_path, 'conv_algo7_128x64_5x5_compute.csv'] );
conv_algo7_128x64_5x5_memory = csvread([saved_model_path, 'conv_algo7_128x64_5x5_memory.csv'] );
conv_algo7_128x64_5x5_under = csvread([saved_model_path, 'conv_algo7_128x64_5x5_under_utilized.csv'] );

if(strcmp(device,'TitanXp'))
    coeff_pre_algo7_128x64_5x5 = csvread([saved_model_path, 'conv_algo7_128x64_5x5_pre.csv']);
end
load([saved_model_path, 'ctree_algo7_128x128_3x3.mat'],'TypeClassificationTree');
ctree_algo7_128x128_3x3 = TypeClassificationTree;
load([saved_model_path, 'rtree_dread_algo7_128x128_3x3.mat'],'DramReadRegTree');
rtree_dread_algo7_128x128_3x3 = DramReadRegTree;
load([saved_model_path, 'rtree_dwrite_algo7_128x128_3x3.mat'],'DramWriteRegTree');
rtree_dwrite_algo7_128x128_3x3 = DramWriteRegTree;
flop_algo7_128x128_3x3 = csvread([saved_model_path, 'flop_algo7_128x128_3x3.csv'] );
conv_algo7_128x128_3x3_compute = csvread([saved_model_path, 'conv_algo7_128x128_3x3_compute.csv'] );
conv_algo7_128x128_3x3_memory = csvread([saved_model_path, 'conv_algo7_128x128_3x3_memory.csv'] );
conv_algo7_128x128_3x3_under = csvread([saved_model_path, 'conv_algo7_128x128_3x3_under_utilized.csv']);

if(strcmp(device,'TitanXp'))
    coeff_pre_algo7_128x128_3x3 = csvread([saved_model_path, 'conv_algo7_128x128_3x3_pre.csv']);
end

load([saved_model_path, 'ctree_algo7_128x128_5x5.mat'],'TypeClassificationTree');
ctree_algo7_128x128_5x5 = TypeClassificationTree;
load([saved_model_path, 'rtree_dread_algo7_128x128_5x5.mat'],'DramReadRegTree');
rtree_dread_algo7_128x128_5x5 = DramReadRegTree;
load([saved_model_path, 'rtree_dwrite_algo7_128x128_5x5.mat'],'DramWriteRegTree');
rtree_dwrite_algo7_128x128_5x5 = DramWriteRegTree;
flop_algo7_128x128_5x5 = csvread([saved_model_path, 'flop_algo7_128x128_5x5.csv'] );
conv_algo7_128x128_5x5_compute = csvread([saved_model_path, 'conv_algo7_128x128_5x5_compute.csv'] );
conv_algo7_128x128_5x5_memory = csvread([saved_model_path, 'conv_algo7_128x128_5x5_memory.csv'] );
conv_algo7_128x128_5x5_under = csvread([saved_model_path, 'conv_algo7_128x128_5x5_under_utilized.csv']);

if(strcmp(device,'TitanXp'))
    coeff_pre_algo7_128x128_5x5 = csvread([saved_model_path, 'conv_algo7_128x128_5x5_pre.csv']);
end

load([saved_model_path, 'ctree_kernel_algo7.mat'],'Mdl');
ctree_kernel_algo7 = Mdl;

%% read test data
% [test_data,~]=xlsread(file_name,1);

batch_size = test_data(:,1);
in_chan = test_data(:,2);
in_wid = test_data(:,3);
out_chan = test_data(:,4);
out_wid = test_data(:,5);
kernel_wid = test_data(:,6);
stride = test_data(:,7);

test_data(:,10) = 1;
predict_time_1 = test_data(:,10)*200000;
predict_time_2 = test_data(:,10)*200000;

%% use coressponding model to predict for each algo, each variant.
% algo7_128x64
% calculate num_blocks
test_data(:,14) = 120;
[row, ~] = size(test_data);
test_data(:,15) = 0;

predict_type_1 = ones(row, 1)*(-1);
predict_type_2 = ones(row, 1)*(-1);

for i = 1 : row
    if(kernel_wid(i) == 3)
        test_data(i,15) = ceil(batch_size(i)*ceil(out_wid(i)/4)^2/128)* ceil(out_chan(i)/64) * 36;
    elseif(kernel_wid(i) == 5)
        test_data(i,15) = ceil(batch_size(i)*ceil(out_wid(i)/9)^2/128)* ceil(out_chan(i)/64) * 169;
    end
end
test_data(:,16) = ceil(test_data(:,15)./test_data(:,14));

for i = 1 : row
    if((kernel_wid(i) == 3 || kernel_wid(i) == 5) && stride(i) == 1)
        % predict type
        X_test = test_data(i, [2 3 4 5 6 7 15]);
        if(kernel_wid(i) == 3)
            [predict_type,~] = predict(ctree_algo7_128x64_3x3,X_test);
        elseif(kernel_wid(i) == 5)
            [predict_type,~] = predict(ctree_algo7_128x64_5x5,X_test);
        end
        test_data(i,19) = predict_type;
        predict_type_1(i) = predict_type;
        % predict # of flop
        test_data(i,20) = 1;
        if(kernel_wid(i) == 3)
            predict_flop_inst_count = flop_func_winograd(flop_algo7_128x64_3x3, test_data(i,:));
        elseif(kernel_wid(i) == 5)
            predict_flop_inst_count = flop_func_winograd(flop_algo7_128x64_5x5, test_data(i,:));
        end
        test_data(i,20) = predict_flop_inst_count;
        % predict # of DRAM
        X_test = test_data(i, [2 3 4 5 6 7]);
        if(kernel_wid(i) == 3)
            [predict_dram_read_ratio,~] = predict(rtree_dread_algo7_128x64_3x3,X_test);
        elseif(kernel_wid(i) == 5)
            [predict_dram_read_ratio,~] = predict(rtree_dread_algo7_128x64_5x5,X_test);
        end    
        test_data(i,23) = predict_dram_read_ratio;
        test_data(i,21) = predict_dram_read_ratio .* in_chan(i);

        if(kernel_wid(i) == 3)
            [predict_dram_write_ratio,~] = predict(rtree_dwrite_algo7_128x64_3x3,X_test);
        elseif(kernel_wid(i) == 5)
            [predict_dram_write_ratio,~] = predict(rtree_dwrite_algo7_128x64_5x5,X_test);
        end    
        test_data(i,24) = predict_dram_write_ratio;
        test_data(i,22) = test_data(i,24);

        % predict overall time
        if(test_data(i,19) == 0)       % compute-bound
            if(kernel_wid(i) == 3)
                predict_time_1(i) = time_compute_bound(conv_algo7_128x64_3x3_compute, test_data(i,:));
            elseif (kernel_wid(i) == 5)
                predict_time_1(i) = time_compute_bound(conv_algo7_128x64_5x5_compute, test_data(i,:));
            end
        else
            if(test_data(i,19) == 1)   % memory-bound
                if(kernel_wid(i) == 3)
                    predict_time_1(i) = time_dram_bound(conv_algo7_128x64_3x3_memory, test_data(i,:));
                elseif (kernel_wid(i) == 5)
                    predict_time_1(i) = time_dram_bound(conv_algo7_128x64_5x5_memory, test_data(i,:));
                end
            else                        
                if(test_data(i,19) == 2)   % under-bound
                    if(kernel_wid(i) == 3)
                        predict_time_1(i) = time_resource_bound(conv_algo7_128x64_3x3_under, test_data(i,:));
                    elseif (kernel_wid(i) == 5)
                        predict_time_1(i) = time_resource_bound(conv_algo7_128x64_5x5_under, test_data(i,:));
                    end
                else
                    fprintf('==ERROR== Unrecognized Kernel Type at row %d\n', i);
                end
            end
        end

        if(strcmp(device,'TitanXp'))
            % predicr pre/post kernels
            if(kernel_wid(i) == 3)
                predict_pre_kernel_time = pre_kernel_algo7(coeff_pre_algo7_128x64_3x3, test_data(i,:));
            elseif (kernel_wid(i) == 5)
                predict_pre_kernel_time = pre_kernel_algo7(coeff_pre_algo7_128x64_5x5, test_data(i,:));
            end
            predict_time_1(i) = predict_time_1(i) + predict_pre_kernel_time;
        end

    end

end

% algo7_128x128
% calculate num_blocks
test_data(:,14) = 60;
[row, ~] = size(test_data);
for i = 1 : row
    if(kernel_wid(i) == 3)
        test_data(i,15) = ceil(batch_size(i)*ceil(out_wid(i)/4)^2/128)* ceil(out_chan(i)/64) * 36;
    elseif(kernel_wid(i) == 5)
        test_data(i,15) = ceil(batch_size(i)*ceil(out_wid(i)/9)^2/128)* ceil(out_chan(i)/64) * 169;
    end
end
test_data(:,16) = ceil(test_data(:,15)./test_data(:,14));

for i = 1 : row
    if((kernel_wid(i) == 3 || kernel_wid(i) == 5) && stride(i) == 1)
        % predict type
        X_test = test_data(i, [2 3 4 5 6 7 15]);
        if(kernel_wid(i) == 3)
            [predict_type,~] = predict(ctree_algo7_128x128_3x3,X_test);
        elseif(kernel_wid(i) == 5)
            [predict_type,~] = predict(ctree_algo7_128x128_5x5,X_test);
        end
        test_data(i,19) = predict_type;
        predict_type_2(i) = predict_type;

        % predict # of flop
        test_data(i,20) = 1;
        if(kernel_wid(i) == 3)
            predict_flop_inst_count = flop_func_winograd(flop_algo7_128x128_3x3, test_data(i,:));
        elseif(kernel_wid(i) == 5)
            predict_flop_inst_count = flop_func_winograd(flop_algo7_128x128_5x5, test_data(i,:));
        end
        test_data(i,20) = predict_flop_inst_count;

        % predict # of DRAM
        X_test = test_data(i, [2 3 4 5 6 7]);
        if(kernel_wid(i) == 3)
            [predict_dram_read_ratio,~] = predict(rtree_dread_algo7_128x128_3x3,X_test);
        elseif(kernel_wid(i) == 5)
            [predict_dram_read_ratio,~] = predict(rtree_dread_algo7_128x128_5x5,X_test);
        end    
        test_data(i,23) = predict_dram_read_ratio;
        test_data(i,21) = predict_dram_read_ratio .* in_chan(i);

        if(kernel_wid(i) == 3)
            [predict_dram_write_ratio,~] = predict(rtree_dwrite_algo7_128x128_3x3,X_test);
        elseif(kernel_wid(i) == 5)
            [predict_dram_write_ratio,~] = predict(rtree_dwrite_algo7_128x128_5x5,X_test);
        end    
        test_data(i,24) = predict_dram_write_ratio;
        test_data(i,22) = test_data(i,24);

        % predict overall time
        if(test_data(i,19) == 0)       % compute-bound
            if(kernel_wid(i) == 3)
                predict_time_2(i) = time_compute_bound(conv_algo7_128x128_3x3_compute, test_data(i,:));
            elseif (kernel_wid(i) == 5)
                predict_time_2(i) = time_compute_bound(conv_algo7_128x128_5x5_compute, test_data(i,:));
            end
        else
            if(test_data(i,19) == 1)   % memory-bound
                if(kernel_wid(i) == 3)
                    predict_time_2(i) = time_dram_bound(conv_algo7_128x128_3x3_memory, test_data(i,:));
                elseif (kernel_wid(i) == 5)
                    predict_time_2(i) = time_dram_bound(conv_algo7_128x128_5x5_memory, test_data(i,:));
                end
            else                        
                if(test_data(i,19) == 2)   % under-bound
                    if(kernel_wid(i) == 3)
                        predict_time_2(i) = time_resource_bound(conv_algo7_128x128_3x3_under, test_data(i,:));
                    elseif (kernel_wid(i) == 5)
                        predict_time_2(i) = time_resource_bound(conv_algo7_128x128_5x5_under, test_data(i,:));
                    end
                else
                    fprintf('==ERROR== Unrecognized Kernel Type at row %d\n', i);
                end
            end
        end

        if(strcmp(device,'TitanXp'))
            % predicr pre/post kernels
            if(kernel_wid(i) == 3)
                predict_pre_kernel_time = pre_kernel_algo7(coeff_pre_algo7_128x128_3x3, test_data(i,:));
            elseif (kernel_wid(i) == 5)
                predict_pre_kernel_time = pre_kernel_algo7(coeff_pre_algo7_128x128_5x5, test_data(i,:));
            end
            predict_time_2(i) = predict_time_2(i) + predict_pre_kernel_time;
        end
    end

end

% predict kernel choice
predict_time_algo7 = predict_time_1;
predict_type_algo7 = [];
[predict_kernel_type,~] = predict(ctree_kernel_algo7, test_data(:,[2 3 4 5 6 7]));
for i = 1: row
    if(predict_kernel_type(i) == 0)
        predict_time_algo7(i) = predict_time_1(i);
        predict_type_algo7(i) = predict_type_1(i);
    elseif(predict_kernel_type(i) == 1)
        predict_time_algo7(i) = predict_time_2(i);
        predict_type_algo7(i) = predict_type_2(i);
    end 
end

test_data(:,10) = predict_time_algo7;
test_data(:,11) = predict_type_algo7;
% csvwrite(['../../final-evaluation-data/',device,'_predict_type_algo7.csv'], test_data);

pred_time = test_data;

% fprintf("Algo7 prediction finished.\n");

end