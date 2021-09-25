import tensorflow as tf
import tensorflow.contrib.slim as slim
import json
from abc import ABCMeta, abstractmethod
from tensorflow.python.ops import math_ops

tmp_files_path = 'data/tmp_results/'

def get_time_from_timeline(file_name, target):
    trace_file = open(file_name, 'r')
    linelist=trace_file.readlines();
    linestrjoin = ''.join(str(e) for e in linelist)
    load_json=json.loads(linestrjoin)

    all_compute_pid = 0
    memcpy_pid = 0
    for i in load_json["traceEvents"]:
        if(i["name"] == "process_name"):
            if(i["args"]["name"] == "/device:GPU:0/stream:all Compute"):
                all_compute_pid = i["pid"]
            if(i["args"]["name"] == "/device:GPU:0/memcpy Compute"):
                memcpy_pid = i["pid"]
    dur = 0
    for i in load_json["traceEvents"]:
        if(i["pid"] == all_compute_pid or i["pid"] == memcpy_pid):
            for key, value in i.items():
                if(key == "cat" and value == "Op"):
                    if(i["args"]["name"] == target):
                        dur = dur + i["dur"]
    return dur/1000.0

class BenchmarkBase(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        self.target = ""
        pass

    @abstractmethod
    def model(self, net):
        pass

    def loss(self, inputs, labels):
        dims = inputs.get_shape().as_list()
        inputs = tf.reshape(inputs, (-1, dims[1] * dims[2] * dims[3]), name='reshape')
        inputs = slim.fully_connected(inputs=inputs, num_outputs=self.num_classes, scope='fc_predict')
        predictions = slim.softmax(inputs, 'predictions')
        loss = slim.losses.softmax_cross_entropy(predictions, labels)
        return loss

    def get_eval_res(self):
        sum_time = 0
        for idx in range(10):
            file_name = tmp_files_path + 'op_timeline_' + str(idx)+'.json'
            sum_time += get_time_from_timeline(file_name, self.target)
        avg_time = sum_time/10
        return avg_time

class BenchmarkConvOP(BenchmarkBase):
    def __init__(self, batch_size=1, num_classes=1, kernel=3, stride=1, in_wid = 1, in_channel=1, out_channel=1, padding='VALID'):
        super(BenchmarkConvOP, self).__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.kernel = kernel
        self.stride = stride
        self.input_chan = in_channel
        self.out_chan = out_channel
        self.target = "conv_benchmark/Conv/Conv2D"
        self.output_shape = []
        self.padding = padding
        self.in_wid = in_wid
        pass

    def model(self, inputs):
        r'''

        :param inputs:
        :return:
        '''
        with tf.variable_scope('conv_benchmark'):

            with tf.device('/gpu:0'):
                inputs = slim.conv2d(inputs = inputs, num_outputs = self.out_chan, kernel_size = self.kernel, stride = self.stride, padding = self.padding)#, normalizer_fn=slim.batch_norm)
                self.output_shape = inputs.get_shape().as_list()
            return inputs

    def get_bn_res(self):
        sum_bn_time = 0
        target = "conv_benchmark/Conv_1/BatchNorm/FusedBatchNorm"
        for idx in range(10):
            file_name = tmp_files_path + 'op_timeline_' + str(idx)+'.json'
            sum_bn_time += get_time_from_timeline(file_name, target)
        avg_bn_time = sum_bn_time/10

        return avg_bn_time    

    def get_relu_res(self):
        sum_relu_time = 0
        target = "conv_benchmark/Conv_1/Relu"
        for idx in range(10):
            file_name = tmp_files_path + 'op_timeline_' + str(idx)+'.json'
            sum_relu_time += get_time_from_timeline(file_name, target)
        avg_relu_time = sum_relu_time/10

        return avg_relu_time                      

    def get_output_shape(self):
        return self.output_shape

class BenchmarkConvOPTrain(BenchmarkBase):
    def __init__(self, batch_size=1, num_classes=1, kernel=3, stride=1, in_wid = 1, in_channel=1, out_channel=1, padding='VALID'):
        super(BenchmarkConvOPTrain, self).__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.kernel = kernel
        self.stride = stride
        self.input_chan = in_channel
        self.out_chan = out_channel
        self.target = "conv_benchmark/Conv_1/Conv2D"
        self.output_shape = []
        self.padding = padding
        self.in_wid = in_wid
        pass

    def model(self, inputs):
        with tf.variable_scope('conv_benchmark'):
            with tf.device('/gpu:0'):
                inputs = slim.conv2d(inputs=inputs, num_outputs=self.input_chan, kernel_size=1, stride=1, biases_initializer=None)
                inputs = slim.conv2d(inputs = inputs, num_outputs = self.out_chan, kernel_size = self.kernel, stride = self.stride, padding = self.padding, normalizer_fn=slim.batch_norm)
                self.output_shape = inputs.get_shape().as_list()
            return inputs

    def get_bn_res(self):
        sum_bn_time = 0
        target = "conv_benchmark/Conv_1/BatchNorm/FusedBatchNorm"
        for idx in range(10):
            file_name = tmp_files_path + 'op_timeline_' + str(idx)+'.json'
            sum_bn_time += get_time_from_timeline(file_name, target)
        avg_bn_time = sum_bn_time/10

        return avg_bn_time    

    def get_relu_res(self):
        sum_relu_time = 0
        target = "conv_benchmark/Conv_1/Relu"
        for idx in range(10):
            file_name = tmp_files_path + 'op_timeline_' + str(idx)+'.json'
            sum_relu_time += get_time_from_timeline(file_name, target)
        avg_relu_time = sum_relu_time/10

        return avg_relu_time                      

    def get_grad_time(self):
        target_grad_input = "gradients/conv_benchmark/Conv_1/Conv2D_grad/Conv2DBackpropInput"
        target_grad_filter = "gradients/conv_benchmark/Conv_1/Conv2D_grad/Conv2DBackpropFilter"
        target_grad_bn = "gradients/conv_benchmark/Conv_1/BatchNorm/FusedBatchNorm_grad/FusedBatchNormGrad"
        target_grad_relu = "gradients/conv_benchmark/Conv_1/Relu_grad/ReluGrad"
        sum_grad_input_time = 0
        sum_grad_filter_time = 0
        sum_grad_bn_time = 0
        sum_grad_relu_time = 0
        for idx in range(10):
            file_name = tmp_files_path + 'op_timeline_' + str(idx)+'.json'
            sum_grad_input_time += get_time_from_timeline(file_name, target_grad_input)
            sum_grad_filter_time += get_time_from_timeline(file_name, target_grad_filter)
            sum_grad_bn_time += get_time_from_timeline(file_name, target_grad_bn)
            sum_grad_relu_time += get_time_from_timeline(file_name, target_grad_relu)
        avg_grad_input_time = sum_grad_input_time/10
        avg_grad_filter_time = sum_grad_filter_time/10
        avg_grad_bn_time = sum_grad_bn_time/10
        avg_grad_relu_time = sum_grad_relu_time/10

        return avg_grad_input_time, avg_grad_filter_time, avg_grad_bn_time, avg_grad_relu_time

    def get_output_shape(self):
        return self.output_shape

class BenchmarkPoolOP(BenchmarkBase):
    def __init__(self, batch_size=1, num_classes=1, kernel=2, stride=2, in_wid = 1, in_channel=1, padding='VALID', pool_fun=slim.avg_pool2d):
        super(BenchmarkPoolOP, self).__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.kernel = kernel
        self.stride = stride
        self.in_wid = in_wid
        self.input_chan = in_channel
        self.padding = padding
        self.pool_fun = pool_fun
        if pool_fun == slim.avg_pool2d:
            self.target = "pool_benchmark/AvgPool2D/AvgPool"
        elif pool_fun == slim.max_pool2d:
            self.target = "pool_benchmark/MaxPool2D/MaxPool"
        self.output_shape = []        
        pass

    def model(self, inputs):
        with tf.variable_scope('pool_benchmark'):
            with tf.device('/gpu:0'):
                inputs = slim.conv2d(inputs=inputs, num_outputs=self.input_chan, kernel_size=1, stride=1)
                inputs = self.pool_fun(inputs=inputs, kernel_size=self.kernel, stride=self.stride, padding='VALID')
                self.output_shape = inputs.get_shape().as_list()         
            return inputs

    def get_grad_time(self):
        if self.pool_fun == slim.avg_pool2d:
            target_grad = "gradients/pool_benchmark/AvgPool2D/AvgPool_grad/AvgPoolGrad"
        elif self.pool_fun == slim.max_pool2d:
            target_grad = "gradients/pool_benchmark/MaxPool2D/MaxPool_grad/MaxPoolGrad"

        sum_grad_time = 0
        for idx in range(10):
            file_name = tmp_files_path + 'op_timeline_' + str(idx)+'.json'
            sum_grad_time += get_time_from_timeline(file_name, target_grad)
        avg_grad_time = sum_grad_time/10
        return avg_grad_time

    def get_output_shape(self):
        return self.output_shape
