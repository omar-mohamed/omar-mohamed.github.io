---
layout:     post
title:      Tensorflow CIFAR-100 Series (7)
author:     Omar Mohamed
tags: 		tensorflow ML CIFAR100 CNN
subtitle:  	ResNets
category:  Technical
---

In [part_6](https://omar-mohamed.github.io/technical/2019/04/20/Tensorflow-CIFAR-100-Series-6(Dropblock)/) we added dropblock support to our
model. In this tutorial we will build a simple resnet upon it and see the results.

## ResNets

There are tons of sources that explain the concept of ResNets and skip connections very well. But very briefly, the idea is that when we build
a deeper neural net we expect the accuracy to keep increasing, but normally this doesn't happen and in fact it worsens. The reason for this is that 
when you go deeper problems like vanishing gradients really start to affect your model's performance. So to solve this problem, skip connections were
added to the network to make it easy to ignore layers that will hurt the performance. So now adding 
more layers will at least not hurt the performance and in better cases make it better.
The following is an image from Andrew Ng's [lecture](https://www.youtube.com/watch?v=ZILIbUvp5lk) on ResNets:

![image](https://user-images.githubusercontent.com/6074821/56459761-af601800-6398-11e9-96ee-61c11cab6d61.png)

So all that we should do is add this skip connections by summing the activations of layer L with the logits of layer L+n(where n can be the skip
length) and feeding this sum to the activation function to get activations of layer L+n. 

![image](https://user-images.githubusercontent.com/6074821/56459813-71172880-6399-11e9-88cf-c8bca992ca3c.png)

## Implementation

First let's see the parameters of our model:

{% highlight python %}
batch_size = 250  # the number of training samples in a single iteration
test_batch_size = 250  # used to calculate test predictions over many iterations to avoid memory issues

drop_block_size = 5  # dropblock size
patch_size_1 = 3  # convolution filter size 1
patch_size_2 = 3  # convolution filter size 2
patch_size_3 = 3  # convolution filter size 2
patch_size_4 = 3  # convolution filter size 2

depth1 = 128  # number of filters in conv block
depth2 = 256  # number of filters in conv block
depth3 = 512  # number of filters in conv block
depth4 = 1024  # number of filters in conv block

num_hidden1 = 4096  # the size of the hidden neurons in fully connected layer
num_hidden2 = 4096  # the size of the hidden neurons in fully connected layer
num_hidden3 = 4096  # the size of the hidden neurons in fully connected layer

{% endhighlight %}

Then we will be updating our convolution block function to use two convolution operations instead of one:

{% highlight python %}

    # method that runs one convolution block with batch normalization
    def run_conv_block(x, layer_name, filter_size, input_depth, output_depth):
        with tf.variable_scope(layer_name):
            conv = run_batch_norm(x)
            conv = tf.nn.relu(conv)
            conv = tf.nn.conv2d(conv,
                                get_conv_weight("weights1", [filter_size, filter_size, input_depth, output_depth]),
                                [1, 1, 1, 1], padding='SAME')
            conv = run_batch_norm(conv)
            conv = tf.nn.relu(conv)
            conv = tf.nn.conv2d(conv,
                                get_conv_weight("weights2", [filter_size, filter_size, output_depth, output_depth]),
                                [1, 1, 1, 1], padding='SAME')

            return conv

{% endhighlight %}

Then we will add another block called a residual block:

{% highlight python %}

    # method that runs one residual block with dropblock and maxbool
    def run_residual_block(x, layer_name, patch_size, input_depth, output_depth):
        with tf.variable_scope(layer_name):
            hidden1 = run_conv_block(x, "conv_block_1", patch_size, input_depth, output_depth)
            hidden1 = tf.nn.relu(hidden1)
            hidden2 = run_conv_block(hidden1, "conv_block_2", patch_size, output_depth, output_depth)
            hidden = tf.nn.relu(hidden1 + hidden2)

            drop_block = DropBlock(keep_prob=conv_keep_prob, block_size=drop_block_size)
            hidden = drop_block(hidden, training=is_training_ph)

            hidden = tf.nn.max_pool(value=hidden, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

            return hidden

{% endhighlight %}

In the above function, all we do is surround the convolution blocks with a new variable scope and run two convolution blocks and sum their values
like we descibed above. Then run dropblock and max bool.

Now let's have a look at the model function now:

{% highlight python %}

    # Model.
    def model(data):
        hidden = normalize_inputs(data)

        hidden = run_residual_block(hidden, "residual_block_1", patch_size_1, num_channels, depth1)

        hidden = run_residual_block(hidden, "residual_block_2", patch_size_2, depth1, depth2)

        hidden = run_residual_block(hidden, "residual_block_3", patch_size_3, depth2, depth3)

        hidden = run_residual_block(hidden, "residual_block_4", patch_size_4, depth3, depth4)

        # flatten
        hidden = tf.contrib.layers.flatten(hidden)

        #  fully connected layers
        hidden = run_hidden_layer(hidden, "fully_connected_1", hidden.shape[1], num_hidden1, fully_connected_keep_prob,
                                  use_activation=True)

        hidden = run_hidden_layer(hidden, "fully_connected_2", num_hidden2, num_hidden3, fully_connected_keep_prob,
                                  use_activation=True)

        hidden = run_hidden_layer(hidden, "fully_connected_3", num_hidden3, images_labels, 1, use_activation=False)

        return hidden

{% endhighlight %}

We will be using 4 residual blocks followed by our three hidden layers.

## Results

Training set accuracy:

![image](https://user-images.githubusercontent.com/6074821/56460395-50070580-63a2-11e9-935f-3fb869fd0f90.png)
![image](https://user-images.githubusercontent.com/6074821/56460400-5ac19a80-63a2-11e9-9aae-d717f55dbf84.png)

Test set accuracy:

![image](https://user-images.githubusercontent.com/6074821/56460401-63b26c00-63a2-11e9-9576-5fdf1491a46e.png)
![image](https://user-images.githubusercontent.com/6074821/56460402-6b721080-63a2-11e9-9367-81e6e5eb1e21.png)

Final training set accuracy: 99.5% <br/>
Final training set loss: 0.0300 <br/>
Final test set accuracy: 70.7% <br/>
Final test set loss: 1.2815 <br/>

So we increased our accuracy with a lower number of training epochs.


If you want to check the full state of the project until now click [here](https://github.com/omar-mohamed/Object-Classification-CIFAR-100) to go the repository. <br/>
See you in part 8.

