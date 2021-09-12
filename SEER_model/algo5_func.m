function [ pred_result ] = algo5_func( beta, x )

a = beta(1);
b = beta(2);
c = beta(3);
d = beta(4);

in_chan = x(:,2);
out_chan = x(:,4);
out_wid = x(:,5);
kernel = x(:,6);

num_kernels = ceil(out_wid./(32-kernel+1)).^2;

target = x(:, 10);

pred_result = (a * in_chan ./target + b * in_chan .* out_chan./target + c * out_chan ./target + d./target).*num_kernels;


end

