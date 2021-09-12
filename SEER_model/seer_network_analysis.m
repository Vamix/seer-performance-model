
device = 'TitanXp';
% saved_model_path = ['test_saved_model_',device,'/'];
% saved_model_path = ['new_saved_model_',device,'/'];
% dataset_path = ['../data/tmp/final_datasets_',device,'/'];
% dataset_path = ['../data/new_profiled_',device,'/'];
dataset_path = ['../data/SEER/src/'];
saved_model_path = ['saved_model_',device,'/'];
suffix = '.xlsx';
data_path = [dataset_path, 'Network-analysis', suffix];


fprintf("Network Analysis Result:\n ");
for index = 1:3
    if (index == 1)
        network_name = "VGG-19";
    end
    if (index == 2)
        network_name = "Inception-V1";
    end
    if (index == 3)
        network_name = "ResNet-50";
    end
    
    [data, ~] = xlsread(data_path,index);
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
    predict_type = [predict_algo0(:,11),predict_algo1(:,11),predict_algo2(:,11),predict_algo3, predict_algo4(:,11),predict_algo5(:,11),predict_algo6(:,11),predict_algo7(:,11)];
    [min_predict_time, min_index]=min(predicted_time,[],2);

    min_index = min_index - 1;

    [num_ops, ~] = size(predicted_time);
    final_type = ones(num_ops,1)* (-2);
    num_compute_bound = 0;
    num_dram_bound = 0;
    num_under_utilized = 0;
    for idx = 1: num_ops
        type_of_current_op = predict_type(idx,min_index(idx)+1);
        if (type_of_current_op == 0)
            num_compute_bound = num_compute_bound + 1;
        end
        if (type_of_current_op == 1)
            num_dram_bound = num_dram_bound + 1;
        end    
        if (type_of_current_op == 2 || type_of_current_op == -2)
            num_under_utilized = num_under_utilized + 1;
        end
        final_type(idx) = predict_type(idx,min_index(idx)+1);
    end

    fprintf("%s[%d]: compute-bound[%d]: %.2f%%, dram-bound[%d]: %.2f%%, under-utilized[%d]: %.2f%%\n ", network_name, num_ops, num_compute_bound, num_compute_bound/num_ops * 100, num_dram_bound, num_dram_bound/num_ops * 100, num_under_utilized, num_under_utilized/num_ops * 100);
end
