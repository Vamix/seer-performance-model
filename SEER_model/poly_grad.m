function [ pred_result ] = poly_grad( beta, x )

a = beta(1);
b = beta(2);

conv_time = x(:, 9);
target = x(:, 12);

pred_result = (a * conv_time + b)./target;

end

