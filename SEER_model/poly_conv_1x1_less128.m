function [ pred_result ] = poly_conv_1x1_less128( beta, x )

a = beta(1);
b = beta(2);
c = beta(3);
d = beta(4);

batch_size = x(:, 1);
in_chan = x(:, 2);
in_wid = x(:, 3);
out_chan = x(:, 4);
out_wid = x(:, 5);
kernel_wid = x(:, 6);
target = x(:, 9);

in_size = batch_size .* in_wid.^2 .* in_chan;
out_size = batch_size .* out_wid.^2 .* out_chan;

mul_times = batch_size .* out_wid.^2 .* out_chan .* kernel_wid.^2 .* in_chan;

pred_result = a * in_size./target + b * out_size./target + c * mul_times./target + d./target;
% pred_result = c * mul_times./target + d./target;

end

