function [ pred_time ] = seer_predict_algo4( device, saved_model_path, test_data )

conv_algo4_16 = csvread([saved_model_path, 'conv_algo4_16.csv']);
conv_algo4_32 = csvread([saved_model_path, 'conv_algo4_32.csv']);
conv_algo4_64 = csvread([saved_model_path, 'conv_algo4_64.csv']);
conv_algo4_128 = csvread([saved_model_path, 'conv_algo4_128.csv']);
conv_algo4_256 = csvread([saved_model_path, 'conv_algo4_256.csv']);

%% read test data
% [test_data,~]=xlsread(file_name,1);
in_wid = test_data(:,3);
stride = test_data(:,7);

%% use coressponding model to predict for each algo, each variant.
test_data(:,10) = 1;
predict_time_algo4 = test_data(:,10)*10000;
[row, ~] = size(test_data);
for i = 1:row   
    if(in_wid(i) <= 256 && stride(i) == 1)
        if(in_wid(i) <= 16)
            predict_time_algo4(i) = algo4_func(conv_algo4_16, test_data(i,:));
        elseif(in_wid(i) <= 32)
            predict_time_algo4(i) = algo4_func(conv_algo4_32, test_data(i,:));
        elseif(in_wid(i) <= 64)
            predict_time_algo4(i) = algo4_func(conv_algo4_64, test_data(i,:));
        elseif(in_wid(i) <= 128)
            predict_time_algo4(i) = algo4_func(conv_algo4_128, test_data(i,:));
        elseif(in_wid(i) <= 256)
            predict_time_algo4(i) = algo4_func(conv_algo4_256, test_data(i,:));
        end
    end
end

test_data(:,10) = predict_time_algo4;
test_data(:,11) = ones(row, 1) *(-1);

% csvwrite(['../../final-evaluation-data/',device,'_predict_type_algo4.csv'], test_data);
pred_time = test_data;

% fprintf("Algo4 prediction finished.\n");

end