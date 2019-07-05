# -*- coding: utf-8 -*-
import math
import torch
import random
import itertools
import numpy as np
import tensorflow as tf

class ACE(tf.keras.Model):

    def __init__(self, dictionary, name='aggregrate_cross_entropy', **kwargs):
        super(ACE, self).__init__(name=name, **kwargs)
        self.softmax = None
        self.label = None
        self.dict = dictionary

    def call(self, inputs):
        input, label = inputs[0], inputs[1]
        self.bs,self.h,self.w,_ = input.shape.as_list()
        T_ = self.h*self.w

        input = tf.reshape(input, (self.bs,T_,-1))
        input = input + 1e-10

        self.softmax = input
        nums,dist = label[:,0],label[:,1:]
        nums = T_ - nums
        
        self.label = tf.concat([tf.expand_dims(nums, -1),dist], 1)

        # ACE Implementation (four fundamental formulas)
        input = tf.reduce_sum(input, axis=1)
        input = input/T_
        label = label/T_
        loss = (-tf.reduce_sum(tf.math.log(input)*label))/self.bs

        return loss


    def decode_batch(self):
        out_best = tf.argmax(self.softmax, 2).numpy()
        pre_result = [0]*self.bs
        for j in range(self.bs):
            pre_result[j] = out_best[j][out_best[j]!=0].astype(np.int32)
        return pre_result


    def vis(self,iteration):
        sn = np.random.randint(0,self.bs-1)
        print(f'Test image {iteration*50+sn:4d}')
        pred = tf.argmax(self.softmax, 2).numpy()
        pred = pred[sn].astype(np.int32).tolist() # sample #0
        pred_string = ''.join([f'{self.dict[pn]:2s}' for pn in pred])
        pred_string_set = [pred_string[i:i+self.w*2] for i in range(0, len(pred_string), self.w*2)]
        print('Prediction: ')
        for pre_str in pred_string_set:
            print(pre_str)
        label = self.label.numpy().astype(np.int32) # (batch_size, num_classes)
        label = ''.join([f'{self.dict[idx]:2s}:{pn:2d}    ' for idx, pn in enumerate(label[sn]) if idx != 0 and pn != 0])
        label = 'Label: ' + label
        print(label)

    def result_analysis(self, iteration):
        prediction = self.decode_batch()
        correct_count = 0
        pre_total = 0
        len_total = self.label[:,1:].numpy().sum()
        label_data = self.label.numpy()
        for idx, pre_list in enumerate(prediction):
            for pw in pre_list:
                if label_data[idx][pw] > 0:
                    correct_count = correct_count + 1
                    label_data[idx][pw] -= 1

            pre_total += len(pre_list)  

        if np.random.random() < 0.05:
            self.vis(iteration)

        return correct_count, len_total, pre_total  