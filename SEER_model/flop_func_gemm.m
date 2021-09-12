function [ pred_result ] = flop_func_gemm( beta, x )

a = beta(1);
b = beta(2);

in_chan = x(:,2);
kernel = x(:,6);
target = x(:, 20);

pred_result = a * kernel.^2 .* in_chan./target + b./target;

end

