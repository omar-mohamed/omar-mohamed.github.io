---
layout:     post
title:      Tensorflow CIFAR-100 Series (2)
author:     Omar Mohamed
tags: 		tensorflow ML CNN CIFAR100
subtitle:  	CIFAR-100 CNN training
category:  Technical
---

In [part_1](https://omar-mohamed.github.io/technical/2019/03/18/Tensorflow-CIFAR-100-Series-1(preprocessing)/) we preprocessed the data saved it in a pickle file for training. In this tutorial we will train a simple CNN and see how it performs.

## Load and reformat the data

This script will begin to download the CIFAR100 dataset in the project folder

{% highlight python %}

from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from matplotlib import pyplot as plt
import random

##################load data#####################

all_data = pickle.load(open('CIFAR_100_processed.pickle', 'rb'))
train_data = all_data['train_dataset']
test_data = all_data['test_dataset']

train_labels = all_data['train_labels']
test_labels = all_data['test_labels']
label_names = all_data['label_names']

del all_data

#################Format train and test data###################


num_channels = 3
image_size = 32


def reformat(dataset):
    dataset = dataset.reshape(
        (-1, image_size, image_size, num_channels)).astype(np.float32)
    return dataset


train_data = reformat(train_data)
test_data = reformat(test_data)

print('train_data shape is : %s' % (train_data.shape,))
print('test_data shape is : %s' % (test_data.shape,))

test_size = test_data.shape[0]
train_size = train_data.shape[0]


############################################################

{% endhighlight %}

Now the data is loaded and formatted in the correct shape for a cnn in tensorflow.


## Model

In this section we will implement the training model. we will begin by adding some constants for the model.

{% highlight python %}

images_labels = 100  # the number of classes
batch_size = 500  # the number of training samples in a single iteration
test_batch_size = 500  # used to calculate test predictions over many iterations to avoid memory issues

patch_size_1 = 7  # convolution filter size 1
patch_size_2 = 5  # convolution filter size 2
patch_size_3 = 3  # convolution filter size 3
patch_size_4 = 3  # convolution filter size 4

depth1 = 64  # number of filters in first conv layer
depth2 = 128  # number of filters in second conv layer
depth3 = 256  # number of filters in third conv layer
depth4 = 512  # number of filters in fourth conv layer

num_hidden1 = 2048  # the size of the hidden neurons in fully connected layer
num_hidden2 = 2048  # the size of the hidden neurons in fully connected layer
num_hidden3 = 2048  # the size of the hidden neurons in fully connected layer

{% endhighlight %}

The batch size is set at 500, but if you run into memory problems you can make lower it down. These constants give you a hint about the model structure with four convolution layers and three fully connected ones.
Now let's start defining our tensorflow graph.

{% highlight python %}


graph = tf.Graph()

with graph.as_default():
    # Input data.
    tf_inputs = tf.placeholder(
        tf.float32, shape=(None, image_size, image_size, num_channels), name='tf_inputs')

    # labels
    tf_labels = tf.placeholder(tf.int32, shape=None, name='tf_labels')

    # dropout keep probability of fully connected layers
    fully_connected_keep_prob = tf.placeholder(tf.float32, name='fully_connected_keep_prob')

    # dropout keep probability of conv layers
    conv_keep_prob = tf.placeholder(tf.float32, name='conv_keep_prob')

    # boolean to determine if in training mode (used in batch norm)
    is_training_ph = tf.placeholder(tf.bool, name='is_training')

{% endhighlight %}

Here we defined our inputs with length None to support any input length, and the same goes for the labels. We will also take as input the fully connected keep probability and the convolution keep probability that will be used when performing dropout to fight overfitting the training set. The final placeholder is a boolean to indicate whether we are training or not, it is used 
for batch normalizaton.

{% highlight python %}
	# a method to normalize the input image to be in range [-1,1]
	def normalize_inputs(inputs):
		pixel_depth = 255.0
		return (inputs - (pixel_depth / 2)) / (pixel_depth / 2)


	# a method to return convolutional weights
	def get_conv_weight(name, shape):
		return tf.get_variable(name, shape=shape,
							   initializer=tf.contrib.layers.xavier_initializer_conv2d())


	# a method to return fully connected weights
	def get_fully_connected_weight(name, shape):
		weights = tf.get_variable(name, shape=shape,
								  initializer=tf.contrib.layers.xavier_initializer())
		return weights

{% endhighlight %}

The first method normalizes the input to be in range [-1,1], the second method returns the convolution weights given a name and a shape (returns the same variable in case same name and shape sent) and third is the same but with fully connected weights.
Now we will begin adding the building blocks of our model.

{% highlight python %}
    def run_batch_norm(inputs):
        return tf.layers.batch_normalization(
            inputs=inputs,
            axis=-1,
            momentum=0.997,
            epsilon=1e-5,
            center=True,
            scale=True,
            training=is_training_ph,
            fused=True
        )


    # method that runs one fully connected layer with batch normalization and dropout
    def run_hidden_layer(x, layer_name, input_size, output_size, keep_dropout_rate=1, use_activation=True):
        with tf.variable_scope(layer_name):
            hidden = tf.matmul(x, get_fully_connected_weight("weights", [input_size, output_size]))

            hidden = run_batch_norm(hidden)

            if use_activation:
                hidden = tf.nn.relu(hidden)
            hidden = tf.nn.dropout(hidden, keep_dropout_rate)
            return hidden

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
            conv = tf.nn.dropout(hidden, conv_keep_prob)

            return conv

{% endhighlight %}

The first method runs batch normalization on a layer. The second runs a fully connected one by multiplying the weights, run batch norm and relu activation(if not last layer). 
The final method runs a conv layer by running batch norm, conv2d, max pool, batch norm, relu, then dropout. It's inspired by the res net block but here we use single conv weights instead of two.
We now have all the building blocks, so let's define our model.

{% highlight python %}
    # Model.
    def model(data):
        hidden = normalize_inputs(data)

        # first conv block
        hidden = run_conv_block(hidden, "conv_block_1", patch_size_1, num_channels, depth1)
        # second conv block
        hidden = run_conv_block(hidden, "conv_block_2", patch_size_2, depth1, depth2)
        # third conv block
        hidden = run_conv_block(hidden, "conv_block_3", patch_size_3, depth2, depth3)
        # fourth conv block
        hidden = run_conv_block(hidden, "conv_block_4", patch_size_4, depth3, depth4)
        
        # flatten
        hidden = tf.contrib.layers.flatten(hidden)

        #  fully connected layers
        hidden = run_hidden_layer(hidden, "fully_connected_1", hidden.shape[1], num_hidden2, fully_connected_keep_prob,
                                  use_activation=True)

        hidden = run_hidden_layer(hidden, "fully_connected_2", num_hidden2, num_hidden3, fully_connected_keep_prob,
                                  use_activation=True)

        hidden = run_hidden_layer(hidden, "fully_connected_3", num_hidden3, images_labels, 1, use_activation=False)

        return hidden
	

{% endhighlight %}

Given an input we first normalize it, run four convolution blocks, then three fully connected ones and get the logits.
Now we compute the loss and update the weights.

{% highlight python %}
    # Training computation.
    logits = model(tf_inputs)

    # loss of softmax with cross entropy
    tf_loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf_labels, logits=logits))

    # for saving batch normalization values
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        # use learning rate decay

        # tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate, staircase=False, name=None)
        # decayed_learning_rate = learning_rate *decay_rate ^ (global_step / decay_steps)
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(0.0005, global_step, train_size/batch_size, 0.99, staircase=True)

        # Optimizer.
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(tf_loss, tvars),
                                          100.0)  # gradient clipping
        optimize = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=global_step)
{% endhighlight %}

We get the loss using cross entropy on softmax of logits. UPDATE_OPS is used for batch normalization to save the values in case you want to predict given a new input. 
We also use learning rate decay and adam optimizer with gradient clipping.
Now we only need to handle inputs than need to be predicted.
{% highlight python %}
    # Predictions for the inputs.

    tf_predictions = tf.nn.softmax(logits)
    tf_predictions = tf.identity(tf_predictions, name='tf_predictions')
{% endhighlight %}

Well that was simple...

## Session

In tensorflow you need to create a session to be able to train your model. So let's do that.

{% highlight python %}

########################Training Session###########################

# computes accuracy given the predictions and real labels
def accuracy(predictions, labels):
    batch_size = predictions.shape[0]
    sum = np.sum(predictions == labels)
    acc = (100.0 * sum) / batch_size
    return acc, predictions
	
num_epochs = 100  # number of training epochs

# used for drawing error and accuracy over time

# augmented training batch
training_batch_loss = []
training_batch_loss_iteration = []
training_batch_accuracy = []
training_batch_accuracy_iteration = []

# normal training set
train_accuracy = []
train_accuracy_iteration = []

train_loss = []
train_loss_iteration = []

# test set
test_accuracy = []
test_accuracy_iteration = []

test_loss = []
test_loss_iteration = []

early_stop_counter = 3  # stop if test loss is not decreasing for early_stop_counter iterations
	
{% endhighlight %}

The first method computes the accuracy of the prediction. We will be training for 100 epoch but we will also be using early stopping in case the loss on test set 
is increasing for 3 epochs. The rest of the values are used for drawing purposes. Now we are ready to begin the session.

{% highlight python %}
with tf.Session(graph=graph, config=tf.ConfigProto(log_device_placement=True)) as session:
    tf.global_variables_initializer().run()
    # to save model after finishing
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter('./graph_info', session.graph)
    step = 0
	
    # method to return accuracy and loss over the sent dataset in steps
    def getAccuracyAndLoss(dataset, labels, batch_size):
        data_size = dataset.shape[0]
        pred = np.zeros((data_size, images_labels))
        overall_loss = 0
        num_iterations = 0
        for offset in range(0, data_size, batch_size):
            batch_labels = labels[offset:(offset + batch_size)]

            batch_data = dataset[offset:(offset + batch_size), :]
            feed_dict = {tf_inputs: batch_data, tf_labels: batch_labels, fully_connected_keep_prob: 1.0,
                         is_training_ph: False,
                         conv_keep_prob: 1.0}
            predictions, l = session.run(
                [tf_predictions, tf_loss], feed_dict=feed_dict)

            pred[offset:offset + batch_size, :] = predictions
            overall_loss = overall_loss + l
            num_iterations = num_iterations + 1
        # calculate accuracy and loss
        overall_loss = overall_loss / num_iterations
        overall_acc, predictions = accuracy(np.argmax(pred, axis=1), labels)
        return overall_acc, overall_loss, predictions
		
    def train_epoch(dataset, labels, batch_size):
        data_size = dataset.shape[0]
        global step
        for offset in range(0, data_size, batch_size):
            batch_data = dataset[offset:(offset + batch_size), :]
            batch_labels = labels[offset:(offset + batch_size)]
            # train on batch and get accuracy and loss
            feed_dict = {tf_inputs: batch_data, tf_labels: batch_labels, fully_connected_keep_prob: 0.5,
                         conv_keep_prob: train_conv_keep_prob, is_training_ph: True}

            _, l, predictions, lr = session.run(
                [optimize, tf_loss, tf_predictions, learning_rate], feed_dict=feed_dict)

            # print results on mini-batch every 5 iteration
            if (step % 25 == 0):
                print('Learning rate at step %d: %.14f' % (step, lr))
                print('DropBlock keep probability at step %d: %.14f' % (step, train_conv_keep_prob))
                print('Minibatch loss at step %d: %f' % (step, l))
                batch_train_accuracy, _ = accuracy(np.argmax(predictions, axis=1), batch_labels)
                print('Minibatch accuracy: %.1f%%' % batch_train_accuracy)
                # save data for plotting
                training_batch_loss.append(l)
                training_batch_loss_iteration.append(step)
                training_batch_accuracy.append(batch_train_accuracy)
                training_batch_accuracy_iteration.append(step)
            step = step + 1		
{% endhighlight %}

Here we start the session and make a method to get accuracy and loss given the data and its labels, and make a method to run one training epoch. 
Now let's use them.

{% highlight python %}

    train_conv_keep_prob = 0.85
    train_conv_keep_prob_min = 0.85 # for linear decrease limit
	train_conv_keep_prob_decrease_per_epoch = 0.005 # for linear decrease after every epoch
    test_predictions = None
    print('Initialized')
    for epoch in range(num_epochs):

        train_epoch(train_data, train_labels, batch_size)

        if train_conv_keep_prob > train_conv_keep_prob_min:
            train_conv_keep_prob = train_conv_keep_prob - train_conv_keep_prob_decrease_per_epoch

        # calculate train loss and accuracy
        overall_train_accuracy, overall_train_loss, _ = getAccuracyAndLoss(train_data, train_labels, batch_size)
        print('train set loss at epoch %d: %f' % (epoch+1, overall_train_loss))
        print('train set accuracy: %.1f%%' % overall_train_accuracy)

        # used for plotting
        train_loss.append(overall_train_loss)
        train_loss_iteration.append(epoch+1)
        train_accuracy.append(overall_train_accuracy)
        train_accuracy_iteration.append(epoch+1)

        # calculate test loss and accuracy
        overall_test_accuracy, overall_test_loss, test_predictions = getAccuracyAndLoss(test_data, test_labels,
                                                                                        test_batch_size)
        print('test set loss at epoch %d: %f' % (epoch+1, overall_test_loss))
        print('test set accuracy: %.1f%%' % overall_test_accuracy)
        # used for plotting
        test_loss.append(overall_test_loss)
        test_loss_iteration.append(epoch+1)
        test_accuracy.append(overall_test_accuracy)
        test_accuracy_iteration.append(epoch+1)


        # early stopping checking
        size = len(test_loss)
        if size > early_stop_counter:
            should_stop = True
            for i in range(early_stop_counter):
                if test_loss[size - 1 - i] <= test_loss[size - 2 - i]:
                    should_stop = False
                    break
            if should_stop:
                print("Early stopping.")
                break

    writer.close()
    saver.save(session, "./saved_model/model.ckpt")
{% endhighlight %}

We run a loop for the number of epochs and after each epoch we calculate loss and accuracy for full training set and test set and save the results for plotting.
We also check if we should early stop and save the model after we finish.
Now let's draw the figures.



{% highlight python %}

###############################Plot Results and save images##############################

# saves accuracy and loss images in folder output_images
def plot_x_y(x, y, figure_name, x_axis_name, y_axis_name, ylim=[0, 100]):
    plt.figure()
    plt.plot(x, y)
    plt.xlabel(x_axis_name)
    plt.ylabel(y_axis_name)
    axes = plt.gca()
    axes.set_ylim(ylim)
    # plt.legend([line_name],loc='upper left')
    plt.savefig('./output_images/' + figure_name)
    plt.show()


plot_x_y(training_batch_loss_iteration, training_batch_loss, 'training_batches_loss.png', 'iteration',
         'augmented training batch loss', [0, 15])
plot_x_y(training_batch_accuracy_iteration, training_batch_accuracy, 'training_batches_acc.png', 'iteration',
         'augmented training batch accuracy')

plot_x_y(train_loss_iteration, train_loss, 'train_loss.png', 'epoch', 'training set loss', [0, 15])
plot_x_y(train_accuracy_iteration, train_accuracy, 'training_acc.png', 'epoch', 'training set accuracy')

plot_x_y(test_loss_iteration, test_loss, 'test_loss.png', 'epoch', 'test set loss', [0, 15])
plot_x_y(test_accuracy_iteration, test_accuracy, 'test_acc.png', 'epoch', 'test set accuracy')



# a method to display and save a sample of the predictions from test set
def disp_prediction_samples(predictions, dataset, num_images, cmap=None):
    for image_num in range(num_images):
        items = random.sample(range(dataset.shape[0]), 8)
        for i, item in enumerate(items):
            plt.subplot(2, 4, i + 1)
            plt.axis('off')
            plt.title(label_names[predictions[item]])
            plt.imshow(np.array(dataset[item, :, :], dtype='uint8'), cmap=cmap, interpolation='none')
        plt.savefig('./output_images/' + 'predictions' + str(image_num + 1) + '.png')
        # plt.show()


disp_prediction_samples(test_predictions, test_data, 10)

print('Final training set accuracy: %.1f%%' % train_accuracy[-1])
print('Final training set loss: %.4f' % train_loss[-1])

print('Final test set accuracy: %.1f%%' % test_accuracy[-1])
print('Final test set loss: %.4f' % test_loss[-1])

{% endhighlight %}

Now we have drawn the plots and also showed a prediction sample. So... What were the results?

## Results

Here are the results of this simple model:

Final training set accuracy: 99.4%
Final training set loss: 0.0377
Final test set accuracy: 56.8%
Final test set loss: 1.8212

![image](https://user-images.githubusercontent.com/6074821/54788263-16be6700-4c37-11e9-9a5d-b20315c59e0c.png)

![image](https://user-images.githubusercontent.com/6074821/54788272-1de57500-4c37-11e9-97c7-57b91e1ec3de.png)

![image](https://user-images.githubusercontent.com/6074821/54788279-2473ec80-4c37-11e9-8f17-daadc16571fe.png)

![image](https://user-images.githubusercontent.com/6074821/54788291-2b026400-4c37-11e9-82d1-47a6e8e2b83d.png)

Not the best thing in the world, unfortunately.. It's apparent that there is a massive case of over fitting here, and it's understandable given that every class only has 500 images to learn from.
And we also note that it stopped after 30 epochs because the test loss was increasing. What can we do to make it perform better on the test set?
Well..lots of things, we are just getting started. In the next tutorial we will see how we can load the saved model and make predictions with it. After that 
we will start to make small but important enhancements to our model to make it generalize better to the test set.
If you want to check the full state of the project until now click [here](https://github.com/omar-mohamed/Object-Classification-CIFAR-100) to go the repository. <br/>
See you in part 3. 
