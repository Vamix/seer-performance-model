function [ pred_result ] = poly_data_layout( beta, x )

a = beta(1);
b = beta(2);

batch_size = x(:, 1);
in_chan = x(:, 2);
in_wid = x(:, 3);

in_size = batch_size .* in_chan .* in_wid.^2;
target = x(:, 4);

pred_result = (a * in_size + b)./target;

end

