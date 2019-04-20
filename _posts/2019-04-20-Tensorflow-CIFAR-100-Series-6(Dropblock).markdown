---
layout:     post
title:      Tensorflow CIFAR-100 Series (6)
author:     Omar Mohamed
tags: 		tensorflow ML CIFAR100
subtitle:  	Dropblocks
category:  Technical
---

In [part_5](https://omar-mohamed.github.io/technical/2019/04/19/Tensorflow-CIFAR-100-Series-5(data-augmentation)/) we used data agmentation to
increase our performance by a large margin. In this tutorial we will introduce a more powerful regularization technique that uses the same idea
of coarse dropout, the data augmentation technique we used, but on the middle layers of the convolutions too instead of normal dropout. 
It's called dropblock.

## DropBlocks

Dropblock builds on the idea of dropout regularization in the case of images to act as a better regularizer. The idea is to drop a whole block 
of neurons instead of a single one in the feature space. The following is an image from the original [paper](https://arxiv.org/abs/1810.12890) describing the idea: 

![image](https://user-images.githubusercontent.com/6074821/56461369-08887580-63b2-11e9-8e75-29f2474a25d1.png)

(a) input image to a convolutional neural network. The green regions in (b) and (c) include
the activation units which contain semantic information in the input image. Dropping out activations
at random is not effective in removing semantic information because nearby activations contain
closely related information. Instead, dropping continuous regions can remove certain semantic
information (e.g., head or feet) and consequently enforcing remaining units to learn features for
classifying input image.

## Implementation

We will be using An Jiaoyang's [implementation](https://github.com/DHZS/tf-dropblock) of dropblock in tensorflow:



{% highlight python %}

# Author: An Jiaoyang
# https://github.com/DHZS/tf-dropblock
# =============================
import tensorflow as tf
from tensorflow.python.keras import backend as K


class DropBlock(tf.keras.layers.Layer):
    def __init__(self, keep_prob, block_size, **kwargs):
        super(DropBlock, self).__init__(**kwargs)
        self.keep_prob = float(keep_prob) if isinstance(keep_prob, int) else keep_prob
        self.block_size = int(block_size)

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        _, self.h, self.w, self.channel = input_shape.as_list()
        # pad the mask
        bottom = right = (self.block_size - 1) // 2
        top = left = (self.block_size - 1) - bottom
        self.padding = [[0, 0], [top, bottom], [left, right], [0, 0]]
        self.set_keep_prob()
        super(DropBlock, self).build(input_shape)

    def call(self, inputs, training=None, scale=True, **kwargs):
        def drop():
            mask = self._create_mask(tf.shape(inputs))
            output = inputs * mask
            output = tf.cond(tf.constant(scale, dtype=tf.bool) if isinstance(scale, bool) else scale,
                             true_fn=lambda: output * tf.to_float(tf.size(mask)) / tf.reduce_sum(mask),
                             false_fn=lambda: output)
            return output

        if training is None:
            training = K.learning_phase()
        output = tf.cond(tf.logical_or(tf.logical_not(training), tf.equal(self.keep_prob, 1.0)),
                         true_fn=lambda: inputs,
                         false_fn=drop)
        return output

    def set_keep_prob(self, keep_prob=None):
        """This method only supports Eager Execution"""
        if keep_prob is not None:
            self.keep_prob = keep_prob
        w, h = tf.to_float(self.w), tf.to_float(self.h)
        self.gamma = (1. - self.keep_prob) * (w * h) / (self.block_size ** 2) / \
                     ((w - self.block_size + 1) * (h - self.block_size + 1))

    def _create_mask(self, input_shape):
        sampling_mask_shape = tf.stack([input_shape[0],
                                       self.h - self.block_size + 1,
                                       self.w - self.block_size + 1,
                                       self.channel])
        mask = DropBlock._bernoulli(sampling_mask_shape, self.gamma)
        mask = tf.pad(mask, self.padding)
        mask = tf.nn.max_pool(mask, [1, self.block_size, self.block_size, 1], [1, 1, 1, 1], 'SAME')
        mask = 1 - mask
        return mask

    @staticmethod
    def _bernoulli(shape, mean):
        return tf.nn.relu(tf.sign(mean - tf.random_uniform(shape, minval=0, maxval=1, dtype=tf.float32)))

{% endhighlight %}

And in our implementation we will change a couple of lines in the convolutional part:

{% highlight python %}

    # method that runs one convolution block
    def run_conv_block(x, layer_name, filter_size, input_depth, output_depth):
        with tf.variable_scope(layer_name):
            conv = run_batch_norm(x)
            conv = tf.nn.conv2d(conv,
                                get_conv_weight("weights", [filter_size, filter_size, input_depth, output_depth]),
                                [1, 1, 1, 1], padding='SAME')
            conv = tf.nn.max_pool(value=conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            conv = run_batch_norm(conv)
            conv = tf.nn.relu(conv)
            drop_block = DropBlock(keep_prob=conv_keep_prob, block_size=drop_block_size)
            conv = drop_block(conv, training=is_training_ph)

            return conv

{% endhighlight %}

drop_block_size is the size of the block to be dropped. It can be initialized at the beginning or changed per layer. Also note that if you
make drop_block_size equal to 1 it will act like normal dropout.

Another thing that was mentioned in the paper when using dropblocks is the decaying of the keep probability over the training iterations for
better divergence. So starting from 1 then decreasing to 0.7 over many iterations might provide you with a better divergence.

DropBlocks are often used with ResNets so in the next tutorial we will extend our model by adding skip connections and see the results of using
a simple ResNet with dropblock.


If you want to check the full state of the project until now click [here](https://github.com/omar-mohamed/Object-Classification-CIFAR-100) to go the repository. <br/>
See you in [part 7](https://omar-mohamed.github.io/technical/2019/04/20/Tensorflow-CIFAR-100-Series-7(ResNet)/).

