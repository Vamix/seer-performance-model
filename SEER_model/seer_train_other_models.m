
device = 'TitanXp';
dataset_path = ['../data/SEER/pre_collected_data_', device, '/'];
saved_model_path = ['saved_model_',device,'/'];
suffix = '.xls';


%% poly model for maxpool ops
file_name = [dataset_path, 'Other-ops-maxpool', suffix];
[train_data_pool,~]=xlsread(file_name,1);
test_data_pool = train_data_pool;
coeff0 = ones(1,4);
[row, ~] = size(train_data_pool);
y_train = ones(row,1);
[coeff_maxpool, ~, ~] = nlinfit(train_data_pool, y_train, @poly_maxpool, coeff0);
csvwrite([saved_model_path, 'maxpool.csv'], coeff_maxpool);    

Y_test = test_data_pool(:,9);
test_data_pool(:,9) = 1;
predict_maxpool_time = poly_maxpool(coeff_maxpool, test_data_pool);

mspe_maxpool = sqrt(mean(((predict_maxpool_time - Y_test)./Y_test) .^ 2));
mse_maxpool = sqrt(mean(((predict_maxpool_time - Y_test)) .^ 2));

fprintf('Maxpool error rate: \t %.4f%%\t %.4f\n', mspe_maxpool*100, mse_maxpool);

%% poly model for backward pool
[train_data_pool,~]=xlsread(file_name,1);
test_data_pool = train_data_pool;
coeff0 = ones(1,3);
[row, ~] = size(train_data_pool);
y_train = ones(row,1);
[coeff_maxpool, ~, ~] = nlinfit(train_data_pool, y_train, @poly_maxpool_grad, coeff0);
csvwrite([saved_model_path, 'maxpool_grad.csv'], coeff_maxpool);    

Y_test = test_data_pool(:,10);
test_data_pool(:,10) = 1;
predict_maxpool_time = poly_maxpool_grad(coeff_maxpool, test_data_pool);

mspe_maxpool = sqrt(mean(((predict_maxpool_time - Y_test)./Y_test) .^ 2));
mse_maxpool = sqrt(mean(((predict_maxpool_time - Y_test)) .^ 2));

fprintf('Maxpool grad error rate: \t %.4f%%\t %.4f\n', mspe_maxpool*100, mse_maxpool);

%% conv 1x1
% for out_chan <=128
file_name = [dataset_path, 'Other-ops-conv_1x1', suffix];
[train_data_conv_1x1,~]=xlsread(file_name,'less128');
% [test_data_conv_1x1,~]=xlsread(file_name,2);
test_data_conv_1x1 = train_data_conv_1x1;

train_data_conv_1x1(:,4) = ceil(train_data_conv_1x1(:,4)/32)*32;
test_data_conv_1x1(:, 4) = ceil(test_data_conv_1x1(:, 4)/32)*32;

coeff0 = ones(1,4);
[row, ~] = size(train_data_conv_1x1);
y_train = ones(row,1);

[coeff_conv_1x1_less128, ~, ~] = nlinfit(train_data_conv_1x1, y_train, @poly_conv_1x1_less128, coeff0);
csvwrite([saved_model_path, 'conv_1x1_less128.csv'], coeff_conv_1x1_less128);    

Y_test = test_data_conv_1x1(:,9);
test_data_conv_1x1(:,9) = 1;
predict_conv1x1_time = poly_conv_1x1_less128(coeff_conv_1x1_less128, test_data_conv_1x1);
mspe_conv_1x1 = sqrt(mean(((predict_conv1x1_time - Y_test)./Y_test) .^ 2));
mse_conv_1x1 = sqrt(mean(((predict_conv1x1_time - Y_test)) .^ 2));
fprintf('Conv_1x1(out_chan <= 128) error rate: \t %.4f%%\t %.4f\n', mspe_conv_1x1*100, mse_conv_1x1);

% for out_chan > 128
[train_data_conv_1x1,~]=xlsread(file_name,'larger128');
% [test_data_conv_1x1,~]=xlsread(file_name,4);
test_data_conv_1x1 = train_data_conv_1x1;
coeff0 = ones(1,2);
[row, ~] = size(train_data_conv_1x1);
y_train = ones(row,1);

[coeff_conv_1x1_larger128, ~, ~] = nlinfit(train_data_conv_1x1, y_train, @poly_conv_1x1_larger128, coeff0);
csvwrite([saved_model_path, 'conv_1x1_larger128.csv'], coeff_conv_1x1_larger128);    

Y_test = test_data_conv_1x1(:,9);
test_data_conv_1x1(:,9) = 1;
predict_conv1x1_time = poly_conv_1x1_larger128(coeff_conv_1x1_larger128, test_data_conv_1x1);
mspe_conv_1x1 = sqrt(mean(((predict_conv1x1_time - Y_test)./Y_test) .^ 2));
mse_conv_1x1 = sqrt(mean(((predict_conv1x1_time - Y_test)) .^ 2));
fprintf('Conv_1x1(out_chan > 128) error rate: \t %.4f%%\t %.4f\n', mspe_conv_1x1*100, mse_conv_1x1);

%% poly model for layout transform
file_name = [dataset_path, 'Other-ops-data_layout', suffix];
[train_data_layout,~]=xlsread(file_name,1);

test_data_layout = train_data_layout;

coeff0 = ones(1,2);
[row, ~] = size(train_data_layout);

train_data_layout_ = [test_data_layout(:,1:3), test_data_layout(:,10)];

y_train = ones(row,1);

[coeff_maxpool, ~, ~] = nlinfit(train_data_layout_, y_train, @poly_data_layout, coeff0);
csvwrite([saved_model_path, 'data_layout.csv'], coeff_maxpool);    

Y_test = test_data_layout(:,10);
test_data_layout_ = [test_data_layout(:, 1:3), ones(row,1)];
predict_maxpool_time = poly_data_layout(coeff_maxpool, test_data_layout_);

mspe_maxpool = sqrt(mean(((predict_maxpool_time - Y_test)./Y_test) .^ 2));
mse_maxpool = sqrt(mean(((predict_maxpool_time - Y_test)) .^ 2));

fprintf('data layout transform error rate: \t %.4f%%\t %.4f\n', mspe_maxpool*100, mse_maxpool);

%% conv grad
file_name = [dataset_path, 'Other-ops-conv_grad', suffix];
[train_data,~]=xlsread(file_name,1);
test_data = train_data;

%% grad input 
X_train = train_data(:, [2 3 4 5 6 7]);
Y_train = train_data(:, 13);
grad_input_rtree = fitrtree(X_train,Y_train, 'PredictorNames',{'in_chan','in_wid','out_chan','out_wid','kernel', 'stride'},'OptimizeHyperparameters','auto','OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',struct('MaxObjectiveEvaluations',30,'ShowPlots',false, 'Verbose', 0), 'MaxNumSplits', 64);
tree_name = [saved_model_path, 'rtree_conv_grad_input.mat'];
save(tree_name,'grad_input_rtree');

X_test = test_data(:, [2 3 4 5 6 7]);
[predict_ratio,~] = predict(grad_input_rtree,X_test);
test_data(:,13) = predict_ratio;

Y_test = test_data(:,10);
predict_grad_input_time = test_data(:,9).* test_data(:,13);

mspe_grad_input = sqrt(mean(((predict_grad_input_time - Y_test)./Y_test) .^ 2));
mse_grad_input = sqrt(mean(((predict_grad_input_time - Y_test)) .^ 2));
fprintf('grad input error rate: \t %.4f%%\t %.4f\n', mspe_grad_input*100, mse_grad_input);

%% grad filter
X_train = train_data(:, [2 3 4 5 6 7]);
Y_train = train_data(:, 14);
grad_filter_rtree = fitrtree(X_train,Y_train, 'PredictorNames',{'in_chan','in_wid','out_chan','out_wid','kernel', 'stride'},'OptimizeHyperparameters','auto','OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',struct('MaxObjectiveEvaluations',30,'ShowPlots',false, 'Verbose', 0), 'MaxNumSplits', 64);
tree_name = [saved_model_path, 'rtree_conv_grad_filter.mat'];
save(tree_name,'grad_filter_rtree');

X_test = test_data(:, [2 3 4 5 6 7]);
[predict_ratio,~] = predict(grad_filter_rtree,X_test);
test_data(:,14) = predict_ratio;

Y_test = test_data(:,11);
predict_grad_filter_time = test_data(:,9).* test_data(:,14);

mspe_grad_filter = sqrt(mean(((predict_grad_filter_time - Y_test)./Y_test) .^ 2));
mse_grad_filter = sqrt(mean(((predict_grad_filter_time - Y_test)) .^ 2));
fprintf('grad filter error rate: \t %.4f%%\t %.4f\n', mspe_grad_filter*100, mse_grad_filter);
