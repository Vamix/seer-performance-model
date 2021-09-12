
device = 'TitanXp';
% saved_model_path = ['test_saved_model_',device,'/'];
% saved_model_path = ['new_saved_model_',device,'/'];
% dataset_path = ['../data/new_profiled_',device,'/'];

dataset_path = ['../data/SEER/pre_collected_data_', device, '/'];
saved_model_path = ['saved_model_',device,'/'];

data_path = [dataset_path, 'Algo-pick-set.xls'];
% predicted_time_save_path = ['test_saved_predicted_time_',device,'/'];

[data, ~] = xlsread(data_path,1);
% [data, ~] = xlsread(data_path,1); % for ResNet
[num_ops, ~] = size(data);

predict_algo0 = seer_predict_algo0(device, saved_model_path, data);
predict_algo1 = seer_predict_algo1(device, saved_model_path, data);
predict_algo2 = seer_predict_algo2(device, saved_model_path, data);
predict_algo4 = seer_predict_algo4(device, saved_model_path, data);
predict_algo5 = seer_predict_algo5(device, saved_model_path, data);
predict_algo6 = seer_predict_algo6(device, saved_model_path, data);
predict_algo7 = seer_predict_algo7(device, saved_model_path, data);

predict_algo3 = ones(num_ops, 1) * 100000;

predicted_time = [predict_algo0(:,10),predict_algo1(:,10),predict_algo2(:,10),predict_algo3, predict_algo4(:,10),predict_algo5(:,10),predict_algo6(:,10),predict_algo7(:,10)];
[min_predict_time, min_index]=min(predicted_time,[],2);

best_algo_index = data(:, 17);
cudnn_pick_algo_index = data(:, 18);
seer_pick_algo_index = min_index - 1;

num_cudnn_false_cases = 0;
num_seer_false_cases = 0;

for i = 1: num_ops
    best_algo_time = data(i, 9 + best_algo_index(i));
    cudnn_pick_time = data(i, 9 + cudnn_pick_algo_index(i));
    seer_pick_time = data(i, 9 + seer_pick_algo_index(i));
    
    if(((cudnn_pick_time - best_algo_time)/best_algo_time) > 0.1)
        num_cudnn_false_cases = num_cudnn_false_cases + 1;
    end
    if(((seer_pick_time - best_algo_time)/best_algo_time) > 0.1)
        num_seer_false_cases = num_seer_false_cases + 1;
    end    
    
end

fprintf("Algo-pick test[total %d cases]\n", num_ops);
fprintf("cuDNN: # wrong cases: %d, error rate: %.4f\n", num_cudnn_false_cases, (num_cudnn_false_cases/num_ops));
fprintf("SEER: # wrong cases: %d, error rate: %.4f\n", num_seer_false_cases, (num_seer_false_cases/num_ops));

