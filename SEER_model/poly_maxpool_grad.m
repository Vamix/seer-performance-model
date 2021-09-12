function [ pred_result ] = poly_maxpool_grad( beta, x )

a = beta(1);
b = beta(2);
c = beta(3);

batch_size = x(:, 1);
in_chan = x(:, 2);
in_wid = x(:, 3);
out_chan = x(:, 4);
out_wid = x(:, 5);
kernel_wid = x(:, 6);
target = x(:, 10);

in_size = batch_size .* in_wid.^2 .* in_chan;
out_size = batch_size .* out_wid.^2 .* out_chan;

pred_result = a * out_size .* kernel_wid./target + b * in_size./target + c ./target;

end

