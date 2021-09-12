function [ pred_time ] = seer_predict_algo5( device, saved_model_path, test_data )

conv_algo5 = csvread([saved_model_path, 'conv_algo5.csv']);
load([saved_model_path,'ctree_algo5.mat'],'TypeClassificationTree');
ctree_algo5 = TypeClassificationTree;

%% read test data
% [test_data,~]=xlsread(file_name,1);
out_wid = test_data(:,5);
kernel_wid = test_data(:,6);
stride = test_data(:,7);

test_data(:,11) = ceil(out_wid./(32-kernel_wid+1)).^2; % num_kernels

%% use coressponding model to predict for each algo, each variant.
test_data(:,10) = 1;
predict_time_algo5 = test_data(:,10)*10000;
[row, ~] = size(test_data);
for i = 1:row   
    if(stride(i) == 1)
        predict_time_algo5(i) = algo5_func(conv_algo5, test_data(i,:));
    end
end

% predict type
X_test = test_data(:, [2 4 5 6]);
[predict_type,~] = predict(ctree_algo5,X_test);
test_data(:,19) = predict_type;

test_data(:,10) = predict_time_algo5;
test_data(:,11) = predict_type;

% csvwrite(['../../final-evaluation-data/',device,'_predict_type_algo5.csv'], test_data);
pred_time = test_data;

% fprintf("Algo5 prediction finished.\n");

end