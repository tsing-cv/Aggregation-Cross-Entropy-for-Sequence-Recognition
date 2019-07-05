# -*- coding: utf-8 -*-
from __future__ import print_function, division
import torch
import argparse
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as KL
from models.ace import ACE
from utils.data_loader import ImageDataset
tf.enable_eager_execution()
# import torch.nn as nn
# from torch import optim
# import torch.nn.functional as F
# from models.seq_module import ACE
# from torch.autograd import Variable
# from models.solver import seq_solver
from utils.basic import timeSince
# from torch.utils.data import DataLoader
# from utils.data_loader import ImageDataset

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='../log/snapshot/model-{:0>2d}.pkl')
parser.add_argument('--total_epoch', type=int, default=50, help='total epoch number')
parser.add_argument('--train_path', type=str, default='../data/train.txt')
parser.add_argument('--test_path', type=str, default='../data/test.txt')
parser.add_argument('--train_batch_size', type=int, default=50, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=50, help='testing batch size')
parser.add_argument('--last_epoch', type=int, default=0, help='last epoch')
parser.add_argument('--class_num', type=int, default=26, help='class number')
parser.add_argument('--dict', type=str, default='_abcdefghijklmnopqrstuvwxyz')
opt = parser.parse_args()

class _Bottleneck(tf.keras.Model):
    def __init__(self, filters, block, 
                 downsampling=False, stride=1, **kwargs):
        super(_Bottleneck, self).__init__(**kwargs)

        filters1, filters2, filters3 = filters
        conv_name_base = 'res' + block + '_branch'
        bn_name_base   = 'bn'  + block + '_branch'

        self.downsampling = downsampling
        self.stride = stride
        self.out_channel = filters3
        
        self.conv2a = KL.Conv2D(filters1, (1, 1), strides=(stride, stride),
                                kernel_initializer='he_normal',
                                name=conv_name_base + '2a')
        self.bn2a = KL.BatchNormalization(name=bn_name_base + '2a')

        self.conv2b = KL.Conv2D(filters2, (3, 3), padding='same',
                                kernel_initializer='he_normal',
                                name=conv_name_base + '2b')
        self.bn2b = KL.BatchNormalization(name=bn_name_base + '2b')

        self.conv2c = KL.Conv2D(filters3, (1, 1),
                                kernel_initializer='he_normal',
                                name=conv_name_base + '2c')
        self.bn2c = KL.BatchNormalization(name=bn_name_base + '2c')
         
        if self.downsampling:
            self.conv_shortcut = KL.Conv2D(filters3, (1, 1), strides=(stride, stride),
                                            kernel_initializer='he_normal',
                                            name=conv_name_base + '1')
            self.bn_shortcut = KL.BatchNormalization(name=bn_name_base + '1')     
    
    def __call__(self, inputs, training=False):
        x = self.conv2a(inputs)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)
        
        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)
        
        x = self.conv2c(x)
        x = self.bn2c(x, training=training)
        
        if self.downsampling:
            shortcut = self.conv_shortcut(inputs)
            shortcut = self.bn_shortcut(shortcut, training=training)
        else:
            shortcut = inputs
            
        x += shortcut
        x = tf.nn.relu(x)
        
        return x     
        

class ResNet(tf.keras.Model):
    def __init__(self, depth, **kwargs):
        super(ResNet, self).__init__(**kwargs)
        if depth not in [50, 101]:
            raise AssertionError('depth must be 50 or 101.')
        self.depth = depth
        self.padding = KL.ZeroPadding2D((3, 3))
        self.conv1 = KL.Conv2D(64, (7, 7), strides=(2, 2), kernel_initializer='he_normal', name='conv1')
        self.bn_conv1 = KL.BatchNormalization(name='bn_conv1')
        self.max_pool = KL.MaxPooling2D((3, 3), strides=(2, 2), padding='same')
        
        self.res2a = _Bottleneck([64, 64, 256], block='2a', downsampling=True, stride=1)
        self.res2b = _Bottleneck([64, 64, 256], block='2b')
        self.res2c = _Bottleneck([64, 64, 256], block='2c')
        
        self.res3a = _Bottleneck([128, 128, 512], block='3a', downsampling=True, stride=2)
        self.res3b = _Bottleneck([128, 128, 512], block='3b')
        self.res3c = _Bottleneck([128, 128, 512], block='3c')
        self.res3d = _Bottleneck([128, 128, 512], block='3d')
        
        self.res4a = _Bottleneck([256, 256, 1024], block='4a', downsampling=True, stride=2)
        self.res4b = _Bottleneck([256, 256, 1024], block='4b')
        self.res4c = _Bottleneck([256, 256, 1024], block='4c')
        self.res4d = _Bottleneck([256, 256, 1024], block='4d')
        self.res4e = _Bottleneck([256, 256, 1024], block='4e')
        self.res4f = _Bottleneck([256, 256, 1024], block='4f')
        if self.depth == 101:
            self.res4g = _Bottleneck([256, 256, 1024], block='4g')
            self.res4h = _Bottleneck([256, 256, 1024], block='4h')
            self.res4i = _Bottleneck([256, 256, 1024], block='4i')
            self.res4j = _Bottleneck([256, 256, 1024], block='4j')
            self.res4k = _Bottleneck([256, 256, 1024], block='4k')
            self.res4l = _Bottleneck([256, 256, 1024], block='4l')
            self.res4m = _Bottleneck([256, 256, 1024], block='4m')
            self.res4n = _Bottleneck([256, 256, 1024], block='4n')
            self.res4o = _Bottleneck([256, 256, 1024], block='4o')
            self.res4p = _Bottleneck([256, 256, 1024], block='4p')
            self.res4q = _Bottleneck([256, 256, 1024], block='4q')
            self.res4r = _Bottleneck([256, 256, 1024], block='4r')
            self.res4s = _Bottleneck([256, 256, 1024], block='4s')
            self.res4t = _Bottleneck([256, 256, 1024], block='4t')
            self.res4u = _Bottleneck([256, 256, 1024], block='4u')
            self.res4v = _Bottleneck([256, 256, 1024], block='4v')
            self.res4w = _Bottleneck([256, 256, 1024], block='4w') 
        
        self.res5a = _Bottleneck([512, 512, 2048], block='5a', downsampling=True, stride=2)
        self.res5b = _Bottleneck([512, 512, 2048], block='5b')
        self.res5c = _Bottleneck([512, 512, 2048], block='5c')
        
        self.out_channel = (256, 512, 1024, 2048)
    
    def __call__(self, inputs, training=True):
        x = self.padding(inputs)
        x = self.conv1(x)
        x = self.bn_conv1(x, training=training)
        x = tf.nn.relu(x)
        x = self.max_pool(x)
        
        x = self.res2a(x, training=training)
        x = self.res2b(x, training=training)
        C2 = x = self.res2c(x, training=training)
        
        x = self.res3a(x, training=training)
        x = self.res3b(x, training=training)
        x = self.res3c(x, training=training)
        C3 = x = self.res3d(x, training=training)
        
        x = self.res4a(x, training=training)
        x = self.res4b(x, training=training)
        x = self.res4c(x, training=training)
        x = self.res4d(x, training=training)
        x = self.res4e(x, training=training)
        x = self.res4f(x, training=training)
        if self.depth == 101:
            x = self.res4g(x, training=training)
            x = self.res4h(x, training=training)
            x = self.res4i(x, training=training)
            x = self.res4j(x, training=training)
            x = self.res4k(x, training=training)
            x = self.res4l(x, training=training)
            x = self.res4m(x, training=training)
            x = self.res4n(x, training=training)
            x = self.res4o(x, training=training)
            x = self.res4p(x, training=training)
            x = self.res4q(x, training=training)
            x = self.res4r(x, training=training)
            x = self.res4s(x, training=training)
            x = self.res4t(x, training=training)
            x = self.res4u(x, training=training)
            x = self.res4v(x, training=training)
            x = self.res4w(x, training=training) 
        C4 = x
        
        # x = self.res5a(x, training=training)
        # x = self.res5b(x, training=training)
        # C5 = x = self.res5c(x, training=training)
        
        # return C2, C3, C4, C5
        return C4

class ResnetEncoderDecoder(tf.keras.Model):
    def __init__(self):
        super(ResnetEncoderDecoder, self).__init__()
        self.resnet = ResNet(50)
        self.out = tf.keras.layers.Dense(opt.class_num+1)
        self.loss_layer = ACE(opt.dict)

    def call(self, inputs, training=True):
        input, label = inputs[0], inputs[1]
        input = self.resnet(input)
        # print ("input shape", input.shape)
        input = tf.nn.softmax(self.out(input),dim=-1)

        return self.loss_layer([input,label])


if __name__ == "__main__":

    model = ResnetEncoderDecoder()


    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001)
    checkpoint_path = "./checkpoints/train"
    ckpt = tf.train.Checkpoint(model=model,
                            optimizer=optimizer,
                            step=tf.train.get_or_create_global_step())
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    train_set = ImageDataset(data_path = opt.train_path, char_path="utils/char.txt", batch_size=128, training=True).data_generation()
    test_set = ImageDataset(data_path = opt.test_path, char_path="utils/char.txt", batch_size=128, training=False).data_generation()

    start_epoch = 0
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
        print (f'Latest checkpoint restored!!\n\tModel path is {ckpt_manager.latest_checkpoint}')

    epochs = 100000
    for epoch in range(start_epoch, epochs):
        loss_history = []
        for step, (inputs, labels) in enumerate(train_set):
            # print (inputs)
            with tf.GradientTape() as tape:
                loss = model(inputs["images"], training=True)
                correct_count, len_total, pre_total = model.loss_layer.result_analysis(step)
                recall = float(correct_count) / len_total
                precision = correct_count / (pre_total+0.000001)
                print(f'Epoch: {epoch:3d} it: {step:6d}, loss: {loss:.4f}, recall: {recall:.4f}, precision: {precision:.4f}')

            grads = tape.gradient(loss, model.variables)
            optimizer.apply_gradients(zip(grads, model.variables), global_step=tf.train.get_or_create_global_step())

            loss_history.append(loss.numpy())
			# if step == 0:
			# 	loss_aver = loss
			# loss_aver = 0.9*loss_aver+0.1*loss			  
			# if step == len(self.lmdb_train)-1:
        ckpt_manager.save()
    # the_solver = seq_solver(model = model,
    #                     lmdb = [lmdb_train, lmdb_test],
    #                     optimizer = optimizer, 
    #                     scheduler = scheduler,
    #                     total_epoch = opt.total_epoch,
    #                     model_path = opt.model_path,
    #                     last_epoch = opt.last_epoch)

    # the_solver.forward()

