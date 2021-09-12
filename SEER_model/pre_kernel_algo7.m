function [ pred_result ] = pre_kernel_algo7( beta, x )

a = beta(1);
b = beta(2);
c = beta(3);
d = beta(4);

target = x(:, 10);

batch_size = x(:, 1);
in_chan = x(:, 2);
in_wid = x(:, 3);
out_chan = x(:, 4);
out_wid = x(:, 5);
kernel_wid = x(:, 6);

in_size1 = batch_size .* in_chan .* in_wid.^2;
in_size2 = batch_size .* in_chan .* out_chan .* kernel_wid.^2;
out_size = batch_size .* out_chan .* out_wid.^2;

pred_result = a * in_size1./target + b * in_size2./target + c * out_size./target+ d./target;

end

