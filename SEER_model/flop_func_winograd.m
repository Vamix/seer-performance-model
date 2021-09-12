function [ pred_result ] = flop_func_winograd( beta, x )

a = beta(1);
b = beta(2);

in_chan = x(:,2);
target = x(:, 20);

% pred_result = a *in_chan./target + b ./target;
pred_result = a *ceil(in_chan/8)./target + b ./target;

end

