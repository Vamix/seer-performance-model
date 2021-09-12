function [ pred_result ] = algo4_func( beta, x )

a = beta(1);
b = beta(2);
c = beta(3);
d = beta(4);

in_chan = x(:,2);
out_chan = x(:,4);
target = x(:, 10);

pred_result = a * in_chan ./target + b * in_chan .* out_chan./target + c * out_chan ./target + d./target;


end

