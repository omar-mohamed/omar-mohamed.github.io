---
layout:     post
title:      Tensorflow CIFAR-100 Series (3)
author:     Omar Mohamed
tags: 		tensorflow ML CIFAR100
subtitle:  	Predicting with the saved model
category:  Technical
---

In [part_2](https://omar-mohamed.github.io/technical/2019/03/22/Tensorflow-CIFAR-100-Series-2(CNN)/) we trained out simple model and saved it in saved_model folder.
In this quick tutorial we will learn how to load a saved model and make classifications with it. We will use the loaded model to classify the test set again.
Just to make sure that it is working well, but of course you can classify any new image given that it is in the correct format which is [num_images, 32,32,3] so in case of single image it will be
[1,32,32,3].

## Load and reformat the test set

If you have read the last part this will seem familiar.

{% highlight python %}

from __future__ import print_function

import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle

##################load data#####################

all_data = pickle.load(open('CIFAR_100_processed.pickle', 'rb'))

test_data = all_data['test_dataset']
test_labels = all_data['test_labels']

del all_data

#################Format test data###################

num_channels = 3
image_size = 32

def reformat(dataset):
    dataset = dataset.reshape(
        (-1, image_size, image_size, num_channels)).astype(np.float32)
    return dataset

test_data = reformat(test_data)

test_size = test_data.shape[0]

{% endhighlight %}

Now the data is loaded and formatted in the correct shape to be fed forward.


## Feed forward

In this section we will load the model and make classifications with it.

{% highlight python %}

# computes accuracy given the predictions and real labels
def accuracy(predictions, labels):
    batch_size = predictions.shape[0]
    sum = np.sum(predictions == labels)
    acc = (100.0 * sum) / batch_size
    return acc, predictions


with tf.Session() as sess:
    #load model
    model_saver = tf.train.import_meta_graph('./saved_model/model.ckpt.meta')
    model_saver.restore(sess, tf.train.latest_checkpoint('./saved_model/'))
    graph = sess.graph
    inputs = graph.get_tensor_by_name("tf_inputs:0")
    fc_keep_prob = graph.get_tensor_by_name("fully_connected_keep_prob:0")
    conv_keep_prob = graph.get_tensor_by_name("conv_keep_prob:0")

    is_training = graph.get_tensor_by_name("is_training:0")
    tf_predictions = graph.get_tensor_by_name("tf_predictions:0")

    # print([node.name for node in graph.as_graph_def().node])

    # get test predictions in steps to avoid memory problems
    test_pred = np.zeros((test_size, images_labels))
    for offset in range(0, test_size, test_batch_size):
        batch_data = test_data[offset:(offset + test_batch_size), :]
        feed_dict = {inputs: batch_data, fc_keep_prob: 1.0, conv_keep_prob: 1.0, is_training: False}
        predictions = sess.run(
            tf_predictions, feed_dict=feed_dict)

        test_pred[offset:offset + test_batch_size, :] = predictions

    # calculate test accuracy
    test_accuracy, test_predictions = accuracy(np.argmax(test_pred, axis=1), test_labels)

    print('Test accuracy: %.1f%%' % test_accuracy)

{% endhighlight %}

All we did in the above code is to load the model, get the tensors we will be feeding or retrieving by name, and going through the test set in batches and making the classifications.
And this concludes this tutorial. In the upcoming ones things will start to get interesting with us trying to enhance the accuracy or training time by using data augmentation, dropblock, res nets, and other cool ideas.
If you want to check the full state of the project until now click [here](https://github.com/omar-mohamed/Object-Classification-CIFAR-100) to go the repository. <br/>
See you in part 4.

