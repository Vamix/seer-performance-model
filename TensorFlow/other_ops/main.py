import os
import benchmark_train 
import multiprocessing
import sys

NUM_CLASSES = 80
BATCH_SIZE = 128

def get_conv_data():

	input_wid_conv = [20, 60, 100, 140, 180]
	input_chan_conv = [50, 150, 250, 350, 450]
	out_chan_conv = [50, 150, 250, 350, 450]
	kernel_conv = [1, 3, 5]				
	stride_conv = [1, 2, 3]		
	# # 	
	# input_wid_conv = [20]#, 60, 100, 140, 180]
	# input_chan_conv = [50]#, 150, 250, 350, 450]
	# out_chan_conv = [50]#, 150, 250, 350, 450]
	# kernel_conv = [1]#, 3, 5]				
	# stride_conv = [1]#, 2, 3]		

	for in_wid in input_wid_conv:
		for in_chan in input_chan_conv:
			for out_chan in out_chan_conv:
				for kernel in kernel_conv:
					for stride in stride_conv:
						print("current: ", in_chan, in_wid, out_chan, kernel, stride)
						# benchmark_train.train_benchmarks_main(op_type='conv_grad', batch_size=BATCH_SIZE, num_classes=NUM_CLASSES, in_wid=in_wid, in_chan=in_chan, out_chan=out_chan, kernel=kernel, stride=stride, padding='SAME', generate_timeline=True)
						p = multiprocessing.Process(target=benchmark_train.train_benchmarks_main, args=('conv_grad',BATCH_SIZE, NUM_CLASSES, in_wid, in_chan, out_chan, kernel, stride, 'SAME', True))
						# p = multiprocessing.Process(target=benchmark_train.train_benchmarks_main, args=('conv',BATCH_SIZE, NUM_CLASSES, in_wid, in_chan, out_chan, kernel, stride, 'SAME', False))
						p.start()
						p.join()

def get_conv_1x1_data():
	print("=====> collecting 1x1 conv data ...")
	input_wid_conv = [10, 20, 40, 60, 80, 100] 
	input_chan_conv = [16, 32, 64, 128, 256, 512, 1024, 2048]
	out_chan_conv = [16, 32, 64, 128, 256, 512, 1024, 2048]

	# input_wid_conv = [10]#, 20, 40, 60, 80, 100] 
	# input_chan_conv = [16]#, 32, 64, 128, 256, 512, 1024, 2048]
	# out_chan_conv = [16]#, 32, 64, 128, 256, 512, 1024, 2048]

	for i in range(len(input_wid_conv)):
		for j in range(len(input_chan_conv)):
			for k in range(len(out_chan_conv)):
				print("========> Training model conv_1x1: ", i,j,k)
				p = multiprocessing.Process(target=benchmark_train.train_benchmarks_main, args=('conv_1x1',BATCH_SIZE, NUM_CLASSES, input_wid_conv[i], input_chan_conv[j], out_chan_conv[k], 1, 1, 'SAME', True))
				p.start()
				p.join()	

def get_avgpool_data():
	print("=====> collecting avgpool data ...")

	input_wid_pool = [20, 60, 100, 140, 180, 220, 260] 	# 7
	input_chan_pool = [50, 150, 250, 350, 450, 550] 	# 7x6 = 42
	kernel_pool = [1, 2, 3, 4, 5]						# 7x6x5 = 210
	stride_pool = [1, 2, 3]								# 7x6x5x3 = 630

	# input_wid_pool = [20]#, 60, 100, 140, 180, 220, 260] 	# 7
	# input_chan_pool = [50]#, 150, 250, 350, 450, 550] 	# 7x6 = 42
	# kernel_pool = [1]#, 2, 3, 4, 5]						# 7x6x5 = 210
	# stride_pool = [1]#, 2, 3]								# 7x6x5x3 = 630	

	for in_wid in input_wid_pool:
		for in_chan in input_chan_pool:
			for kernel in kernel_pool:
				for stride in stride_pool:
					p = multiprocessing.Process(target=benchmark_train.train_benchmarks_main, args=('avgpool',BATCH_SIZE, NUM_CLASSES, in_wid, in_chan, 0, kernel, stride, 'VALID', True))
					p.start()
					p.join()	

def get_maxpool_data():
	print("=====> collecting maxpool data ...")
	input_wid_pool = [20, 60, 100, 140, 180, 220, 260] 	# 7
	input_chan_pool = [50, 150, 250, 350, 450, 550] 	# 7x6 = 42
	kernel_pool = [1, 2, 3, 4, 5]						# 7x6x5 = 210
	stride_pool = [1, 2, 3]								# 7x6x5x3 = 630

	# input_wid_pool = [20]#, 60, 100, 140, 180, 220, 260] 	# 7
	# input_chan_pool = [50]#, 150, 250, 350, 450, 550] 	# 7x6 = 42
	# kernel_pool = [1]#, 2, 3, 4, 5]						# 7x6x5 = 210
	# stride_pool = [1]#, 2, 3]								# 7x6x5x3 = 630

	for in_wid in input_wid_pool:
		for in_chan in input_chan_pool:
			for kernel in kernel_pool:
				for stride in stride_pool:
					p = multiprocessing.Process(target=benchmark_train.train_benchmarks_main, args=('maxpool',BATCH_SIZE, NUM_CLASSES, in_wid, in_chan, 0, kernel, stride, 'VALID', True))
					p.start()
					p.join()
					
def get_layout_trans_data():
	# small tiles
	# for inwid in range(5, 16, 5): # x3
	# 	for inchan in range(30, 301, 30): # 3x10
	# for inwid in range(5, 6, 5): # x3
	# 	for inchan in range(30, 31, 30): # 3x10	
	# 		# for ker_wid in range(3, 10, 2): # x4
	# 		benchmark_train.train_benchmarks_main(op_type='layout', in_wid=inwid, in_chan=inchan, out_chan=inchan, kernel=3, stride=1, padding='SAME', generate_timeline=False)
	# using tiles
	for inwid in [10, 40, 80, 120, 160, 200, 240, 280, 320]: 
		for inchan in [16, 32, 64, 128, 256, 400, 512]: 		
	# for inwid in [10]:#, 40, 80, 120, 160, 200, 240, 280, 320]: 
	# 	for inchan in [16]:#, 32, 64, 128, 256, 400, 512]: 		 	
			# for ker_wid in range(3, 10, 2): # x4
			benchmark_train.train_benchmarks_main(op_type='layout', batch_size=128, in_wid=inwid, in_chan=inchan, out_chan=inchan, kernel=3, stride=1, padding='SAME', generate_timeline=False)

if __name__ == "__main__":

	if(len(sys.argv) > 1):
		file_name = sys.argv[1]
		if file_name == "maxpool":
			get_maxpool_data()
		elif file_name == "conv_1x1":
			get_conv_1x1_data()
		elif file_name == "conv_grad":
			get_conv_data()
		elif file_name == "layout":
			get_layout_trans_data()			
		else:
			with open(file_name) as src_data:
				tmp_lines = src_data.readlines()
				num_ops = int(tmp_lines[0])
				for index in range(1, num_ops+1):
					hyper_params = tmp_lines[index].split("\n")[0].split("\t")
					if(len(hyper_params) >=8):
						batch_size = int(hyper_params[0])
						in_chan = int(hyper_params[1])
						in_wid = int(hyper_params[2])
						out_chan = int(hyper_params[3])
						out_wid = int(hyper_params[4])
						kernel = int(hyper_params[5])
						stride = int(hyper_params[6])
						pad_wid = int(hyper_params[7])
						print("current params: ", batch_size, in_chan, in_wid, out_chan, out_wid, kernel, stride, pad_wid)
						if(pad_wid > 0):
							benchmark_train.train_benchmarks_main('conv', batch_size, 10, in_wid, in_chan, out_chan, kernel, stride, 'SAME', False)
						else:
							benchmark_train.train_benchmarks_main('conv', batch_size, 10, in_wid, in_chan, out_chan, kernel, stride, 'VALID', False)
	else:
		print("=====> [ERROR] Need an argument, can be one of 'maxpool', 'conv_1x1', 'conv_train' 'layout' or a file name")