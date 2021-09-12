function [ pred_time ] = time_compute_bound( beta, x )

a = beta(1);
b = beta(2);

time = x(:, 10);
block_max = x(:, 14);
iter = x(:,16);
inst_fp_blk = x(:, 20);

pred_time = (iter.*(a * block_max .* inst_fp_blk + b))./time;

end

