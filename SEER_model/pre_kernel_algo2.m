function [ pred_result ] = pre_kernel_algo2( beta, x )

a = beta(1);
b = beta(2);
c = beta(3);

batch_size = x(:, 1);
in_chan = x(:, 2);
out_chan = x(:, 4);
out_wid = x(:, 5);
kernel_wid = x(:, 6);
target = x(:, 10);
mid_kernel1 = batch_size .* out_wid.^2 .* kernel_wid.^2 .* in_chan;
mid_kernel2 = out_chan.* kernel_wid.^2 .* in_chan;

pred_result = a * mid_kernel1./target + b * mid_kernel2./target + c ./target;

end

