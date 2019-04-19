---
layout:     post
title:      Tensorflow CIFAR-100 Series (4)
author:     Omar Mohamed
tags: 		tensorflow ML CIFAR100 CNN
subtitle:  	Depth-wise separable convolution
category:  Technical
---

In [part_2](https://omar-mohamed.github.io/technical/2019/03/22/Tensorflow-CIFAR-100-Series-2(CNN)/) we trained our simple model and saved it in saved_model folder.
But as you may have noticed, training a convolutional network takes a lot of time and resources. That is why in this tutorial we will introduce the concept of separable convolution, which is faster than normal convolution, and see how it compares. 

## Depth-wise separable convolution

Depth-wise separable convolution consists of two basic ideas: depth-wise convolution and point-wise convolution.

### Depth-wise convolution

Let us first talk about the idea of depth-wise convolution. The idea is very simple, instead of the filters being of the same depth as the input 
you make a filter for every channel.

![image](https://user-images.githubusercontent.com/6074821/56434510-2af89100-62d5-11e9-864f-918e68583ab7.png)

This is an image of a normal convolution with 256 filters. Now let's see what depth-wise convolution looks like:

![image](https://user-images.githubusercontent.com/6074821/56434558-5f6c4d00-62d5-11e9-9e8c-737b45d8fa03.png)

A filter for every dimension like we discussed earlier but what if we want to change the depth or increase it? That's where point-wise convolution comes in.

### Point-wise convolution

Point-wise convolution is a 1x1xD filter that is used after depth-wise convolution.

![image](https://user-images.githubusercontent.com/6074821/56434894-d0603480-62d6-11e9-91f8-70a8d806a69b.png)

The image above uses 256 of these filters to get to the same shape of the normal convolution in the first image.

So this is separable depth-wise convolution briefly. If you want more information on why it is faster check out this link [A Basic Introduction to Separable Convolutions](https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728)


## Implementation

Only one line in our whole implementation of [part 2](https://omar-mohamed.github.io/technical/2019/03/22/Tensorflow-CIFAR-100-Series-2(CNN)/) will change:


{% highlight python %}

    # method that runs one convolution block
    def run_conv_block(x, layer_name, filter_size, input_depth, output_depth):
        with tf.variable_scope(layer_name):
            conv = run_batch_norm(x)
            conv = tf.nn.separable_conv2d(conv,
                                          get_conv_weight("depth_wise", [filter_size, filter_size, input_depth, 1]),
                                          get_conv_weight("point_wise", [1, 1, input_depth, output_depth]),
                                          strides=[1, 1, 1, 1], padding='SAME')
            conv = tf.nn.max_pool(value=conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            conv = run_batch_norm(conv)
            conv = tf.nn.relu(conv)
            conv = tf.nn.dropout(hidden, conv_keep_prob)

            return conv


{% endhighlight %}

And that's it.

## Results

Training set accuracy:

![image](https://user-images.githubusercontent.com/6074821/56436735-2df77f80-62dd-11e9-91fd-faed46d17b14.png)

Test set accuracy:

![image](https://user-images.githubusercontent.com/6074821/56436777-48c9f400-62dd-11e9-82d1-12f80e68e07c.png)

Final training set accuracy: 99.4% <br/>
Final training set loss: 0.0448 <br/>
Final test set accuracy: 51.3% <br/>
Final test set loss: 2.1067 <br/>

The results are not better than normal convolution, but close nonetheless and much faster.



If you want to check the full state of the project until now click [here](https://github.com/omar-mohamed/Object-Classification-CIFAR-100) to go the repository. <br/>
See you in [part 5](https://omar-mohamed.github.io/technical/2019/04/19/Tensorflow-CIFAR-100-Series-5(Data-augmentation)/).

