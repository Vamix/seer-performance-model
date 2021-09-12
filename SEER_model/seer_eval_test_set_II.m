
device = 'TitanXp';

dataset_path = ['../data/SEER/pre_collected_data_', device, '/'];
saved_model_path = ['saved_model_',device,'/'];
suffix = '.xls';

data_path = [dataset_path, 'conv-test-set-II', suffix];

[data, ~] = xlsread(data_path,1);
[num_ops, ~] = size(data);
real_time = data(:, 10);

predict_algo0 = seer_predict_algo0(device, saved_model_path, data);
predict_algo1 = seer_predict_algo1(device, saved_model_path, data);
predict_algo2 = seer_predict_algo2(device, saved_model_path, data);
predict_algo3 = ones(num_ops, 1) * 100000;
predict_algo4 = seer_predict_algo4(device, saved_model_path, data);
predict_algo5 = seer_predict_algo5(device, saved_model_path, data);
predict_algo6 = seer_predict_algo6(device, saved_model_path, data);
predict_algo7 = seer_predict_algo7(device, saved_model_path, data);

predicted_time = [predict_algo0(:,10),predict_algo1(:,10),predict_algo2(:,10),predict_algo3, predict_algo4(:,10),predict_algo5(:,10),predict_algo6(:,10),predict_algo7(:,10)];
predict_type = [predict_algo0(:,11),predict_algo1(:,11),predict_algo2(:,11),predict_algo3, predict_algo4(:,11),predict_algo5(:,11),predict_algo6(:,11),predict_algo7(:,11)];
[min_predict_time, ~]=min(predicted_time,[],2);

mspe_overall = sqrt(mean(((min_predict_time - real_time)./real_time) .^ 2));
mse_overall = sqrt(mean(((min_predict_time - real_time)) .^ 2));
fprintf('[Test-set-II] Overall error : %.4f%%\t %.4f\n', mspe_overall*100, mse_overall);
