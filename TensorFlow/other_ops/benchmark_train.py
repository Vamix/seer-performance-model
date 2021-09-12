import os
import sys
import time
import xlrd
import xlwt
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.client import timeline
from xlwt import easyxf, Workbook
from xlutils.copy import copy as xl_copy
from benchmark_ops import *

slim = tf.contrib.slim

save_path = 'data/profiled/'
tmp_files_path = 'data/tmp_results/'

LEARNING_RATE = 0.01

def append_reference_data(op_type, batch_size, in_wid, in_chan, out_wid, out_chan, kernel_wid, stride, padding_wid, run_time, optional1, optional2, optional3, optional4, optional5, optional6):
    file_name = save_path + 'Other-ops-'+str(op_type)+'.xls'
    optional1_name = 'optional1'
    optional2_name = 'optional2'
    optional3_name = 'optional3'
    optional4_name = 'optional4'
    optional5_name = 'optional5'
    optional6_name = 'optional6'
    if op_type == 'conv_grad':
        optional1_name = 'grad_input_time'
        optional2_name = 'grad_filter_time'
        optional3_name = 'sum_grad'
        optional4_name = 'ratio_1'
        optional5_name = 'ratio_2'
        optional3 = optional1 + optional2
        optional4 = optional1/run_time
        optional5 = optional2/run_time
    elif op_type == 'avgpool' or 'maxpool':
        optional1_name = 'grad_time'

    try:
        workbook_src = xlrd.open_workbook(file_name)
        sheet_src = workbook_src.sheet_by_index(0)
        nrows = sheet_src.nrows
        workbook = xl_copy(workbook_src)
        sheet = workbook.get_sheet(0)
        sheet.write(nrows, 0, 0)
        sheet.write(nrows, 1, batch_size)
        sheet.write(nrows, 2, in_chan)
        sheet.write(nrows, 3, in_wid)
        sheet.write(nrows, 4, out_chan)
        sheet.write(nrows, 5, out_wid)   
        sheet.write(nrows, 6, kernel_wid)             
        sheet.write(nrows, 7, stride)
        sheet.write(nrows, 8, padding_wid)
        sheet.write(nrows, 9, run_time)   
        sheet.write(nrows, 10, optional1)
        sheet.write(nrows, 11, optional2)
        sheet.write(nrows, 12, optional3)
        sheet.write(nrows, 13, optional4)        
        sheet.write(nrows, 14, optional5)
        sheet.write(nrows, 15, optional6)                  
        workbook.save(file_name)
    except Exception as e:
        print("=========DEBUG:", e)
        workbook = xlwt.Workbook()
        sheet = workbook.add_sheet('new')
        sheet.write(0, 0, 'op_type')
        sheet.write(0, 1, 'batch_size')
        sheet.write(0, 2, 'in_chan')
        sheet.write(0, 3, 'in_wid')
        sheet.write(0, 4, 'out_chan')
        sheet.write(0, 5, 'out_wid')   
        sheet.write(0, 6, 'ker_wid')             
        sheet.write(0, 7, 'stride')
        sheet.write(0, 8, 'padding_wid')
        sheet.write(0, 9, 'run_time')
        sheet.write(0, 10, optional1_name)
        sheet.write(0, 11, optional2_name)
        sheet.write(0, 12, optional3_name)
        sheet.write(0, 13, optional4_name)        
        sheet.write(0, 14, optional5_name)
        sheet.write(0, 15, optional6_name)     
        sheet.write(1, 0, 0)
        sheet.write(1, 1, batch_size)
        sheet.write(1, 2, in_chan)
        sheet.write(1, 3, in_wid)
        sheet.write(1, 4, out_chan)
        sheet.write(1, 5, out_wid)   
        sheet.write(1, 6, kernel_wid)             
        sheet.write(1, 7, stride)
        sheet.write(1, 8, padding_wid)
        sheet.write(1, 9, run_time)      
        sheet.write(1, 10, optional1)
        sheet.write(1, 11, optional2)
        sheet.write(1, 12, optional3)
        sheet.write(1, 13, optional4)        
        sheet.write(1, 14, optional5)
        sheet.write(1, 15, optional6)          
        workbook.save(file_name)

def append_reference_data_conv_1x1(op_type, batch_size, in_chan, in_wid, out_chan, out_wid, kernel_wid, stride, padding_wid, run_time):
    file_name = save_path + 'Other-ops-conv_1x1.xls'
    try:
        workbook_src = xlrd.open_workbook(file_name)
        workbook = xl_copy(workbook_src)
        if out_chan <= 128:
            sheet_src = workbook_src.sheet_by_name('less128')
            sheet = workbook.get_sheet(0)
        else:
            sheet_src = workbook_src.sheet_by_name('larger128')
            sheet = workbook.get_sheet(1)
        nrows = sheet_src.nrows

        sheet.write(nrows, 0, 0)
        sheet.write(nrows, 1, batch_size)
        sheet.write(nrows, 2, in_chan)
        sheet.write(nrows, 3, in_wid)
        sheet.write(nrows, 4, out_chan)
        sheet.write(nrows, 5, out_wid)   
        sheet.write(nrows, 6, kernel_wid)             
        sheet.write(nrows, 7, stride)
        sheet.write(nrows, 8, padding_wid)
        sheet.write(nrows, 9, run_time)               
        workbook.save(file_name)
    except Exception as e:
        print("=========DEBUG:", e)
        workbook = xlwt.Workbook()
        sheet0 = workbook.add_sheet('less128')
        sheet0.write(0, 0, 'op_type')
        sheet0.write(0, 1, 'batch_size')
        sheet0.write(0, 2, 'in_chan')
        sheet0.write(0, 3, 'in_wid')
        sheet0.write(0, 4, 'out_chan')
        sheet0.write(0, 5, 'out_wid')   
        sheet0.write(0, 6, 'ker_wid')             
        sheet0.write(0, 7, 'stride')
        sheet0.write(0, 8, 'padding_wid')
        sheet0.write(0, 9, 'run_time')        
        sheet1 = workbook.add_sheet('larger128')
        sheet1.write(0, 0, 'op_type')
        sheet1.write(0, 1, 'batch_size')
        sheet1.write(0, 2, 'in_chan')
        sheet1.write(0, 3, 'in_wid')
        sheet1.write(0, 4, 'out_chan')
        sheet1.write(0, 5, 'out_wid')   
        sheet1.write(0, 6, 'ker_wid')             
        sheet1.write(0, 7, 'stride')
        sheet1.write(0, 8, 'padding_wid')
        sheet1.write(0, 9, 'run_time')   

        if out_chan <=128:
            target_sheet = sheet0
        else:
            target_sheet = sheet1
        target_sheet.write(1, 0, 0)
        target_sheet.write(1, 1, batch_size)
        target_sheet.write(1, 2, in_chan )
        target_sheet.write(1, 3, in_wid)
        target_sheet.write(1, 4, out_chan )
        target_sheet.write(1, 5, out_wid)   
        target_sheet.write(1, 6, kernel_wid)             
        target_sheet.write(1, 7, stride)
        target_sheet.write(1, 8, padding_wid)
        target_sheet.write(1, 9, run_time)     

        workbook.save(file_name)

def train_model(benchmark):
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        images = tf.random_uniform(shape = [benchmark.batch_size, benchmark.in_wid, benchmark.in_wid, benchmark.input_chan], minval=0, maxval=256, dtype=tf.float32, seed=None, name=None)
        labels = tf.random_uniform(shape = [benchmark.batch_size,], minval=0, maxval=10000, dtype=tf.int32, seed=None, name=None)
        labels = slim.one_hot_encoding(labels, benchmark.num_classes)
        logits = benchmark.model(images)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            iters = 1
            for idx in range(iters):
                sess.run(logits)

def train_model_generate_tl(benchmark):
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        images = tf.random_uniform(shape = [benchmark.batch_size, benchmark.in_wid, benchmark.in_wid, benchmark.input_chan], minval=0, maxval=256, dtype=tf.float32, seed=None, name=None)
        labels = tf.random_uniform(shape = [benchmark.batch_size,], minval=0, maxval=10000, dtype=tf.int32, seed=None, name=None)
        labels = slim.one_hot_encoding(labels, benchmark.num_classes)
        logits = benchmark.model(images)
        loss = benchmark.loss(logits, labels)
        train_op = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            iters = 15
            skip_iter = 5
            for idx in range(iters):
                sess.run([train_op], options=options, run_metadata=run_metadata)

                if idx >= skip_iter:
                    trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                    file_name = tmp_files_path + 'op_timeline_'+str(idx - skip_iter)+'.json'
                    trace_file = open(file_name, 'w')
                    trace_file.write(trace.generate_chrome_trace_format(show_memory=False))  
                    trace_file.close()

def benchmarks_configs(batch_size, num_classes, op_type, in_wid, in_chan, out_chan, kernel, stride, padding):
    if op_type == 'conv' or op_type == 'conv_1x1' or op_type == 'layout':               
        benchmark = BenchmarkConvOP(batch_size, num_classes, kernel=kernel, stride=stride, in_wid=in_wid, in_channel = in_chan, out_channel = out_chan, padding = padding)
    elif op_type == 'conv_grad':
        benchmark = BenchmarkConvOPTrain(batch_size, num_classes, kernel=kernel, stride=stride, in_wid=in_wid, in_channel = in_chan, out_channel = out_chan, padding = padding)
    elif op_type == 'maxpool':
        benchmark = BenchmarkPoolOP(batch_size, num_classes, kernel, stride, in_wid, in_chan, padding, slim.max_pool2d)
    return benchmark


def train_benchmarks_main(op_type, batch_size=1, num_classes=10, in_wid=0, in_chan=0, out_chan=0, kernel=0, stride=0, padding='VALID', generate_timeline=True, index=0):
    padding_wid = 0
    if padding == 'VALID':
        out_wid = int((in_wid - kernel)/stride) + 1
    else:
        out_wid = int((in_wid + stride - 1)/stride)
        padding_wid = (out_wid - 1) * stride + kernel - in_wid
    size_in_pixel = in_wid * in_wid * in_chan
    if op_type == 'maxpool' or op_type == 'avgpool':
        size_out_pixel = out_wid * out_wid * in_chan
    else:
        size_out_pixel = out_wid * out_wid * out_chan
    size_in = (batch_size * size_in_pixel * 4.0) / (1024*1024*1024)
    size_out = (batch_size * size_out_pixel * 4.0) / (1024*1024*1024)

    if (size_in + size_out) > 4:
        return

    benchmark = benchmarks_configs(batch_size, num_classes, op_type, in_wid, in_chan, out_chan, kernel, stride, padding)
    run_time = -1

    if generate_timeline == False:
        train_model(benchmark)
        print("%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d" % (batch_size, in_chan, in_wid, out_chan, out_wid, kernel, stride, 0, -1))
    else:
        train_model_generate_tl(benchmark)
        run_time = benchmark.get_eval_res()
        print("=====> Time By Timeline: ", run_time)
        if op_type == 'layout':
            output_shape = benchmark.get_output_shape()
            out_wid = output_shape[1]
            append_reference_data('layout', batch_size, in_wid, in_chan, out_wid, out_chan, kernel, stride, padding_wid, run_time, 0, 0, 0, 0, 0, 0)      
        elif op_type == 'conv_grad':
            grad_input_time, grad_filter_time, grad_bn_time, grad_relu_time = benchmark.get_grad_time()
            append_reference_data('conv_grad', batch_size, in_wid, in_chan, out_wid, out_chan, kernel, stride, padding_wid, run_time, grad_input_time, grad_filter_time, 0, 0, 0, 0)  
        elif op_type == 'maxpool':
            output_shape = benchmark.get_output_shape()
            out_wid = output_shape[1]
            grad_time = benchmark.get_grad_time()
            append_reference_data('maxpool', batch_size, in_wid, in_chan, out_wid, in_chan, kernel, stride, padding_wid, run_time, grad_time, 0, 0, 0, 0, 0)  
        elif op_type == 'conv_1x1':
            output_shape = benchmark.get_output_shape()
            out_wid = output_shape[1]
            append_reference_data_conv_1x1('conv_1x1', batch_size, in_chan, in_wid, out_chan, out_wid, kernel, stride, padding_wid, run_time, index)  
        else:
            print("====ERROR: unsupported op_type: ", op_type)


if __name__ == "__main__":
    op_type = sys.argv[1]
    batch_size = int(sys.argv[2])
    num_classes = int(sys.argv[3])
    input_wid = int(sys.argv[4])
    input_chan = int(sys.argv[5])
    out_chan = int(sys.argv[6])
    kernel = int(sys.argv[7])
    stride = int(sys.argv[8])
    padding = sys.argv[9]
    generate_timeline = sys.argv[10]
    train_benchmarks_main(op_type, batch_size, num_classes, input_wid, input_chan, out_chan, kernel, stride, padding, generate_timeline)
