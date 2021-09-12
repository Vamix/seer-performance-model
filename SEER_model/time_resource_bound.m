function [ pred_time ] = time_resource_bound( beta, x )

a = beta(1);
b = beta(2);
c = beta(3);
d = beta(4);

time = x(:, 10);
block_max = x(:, 14);
block_launch = x(:, 15);
iter = x(:,16);
inst_fp_blk = x(:, 20);
dram_read_blk = x(:, 21);
dram_write_blk = x(:, 22);
% dram_blk = dram_read_blk + dram_write_blk;

block_last_wave = block_launch - (iter - 1).* block_max;
epsilon = block_last_wave ./ block_max;

pred_time = ((iter - 1 + epsilon).*(a * block_max .* dram_read_blk + d * block_max .* dram_write_blk) + b * iter .* block_max .* inst_fp_blk + c)./time;

end

