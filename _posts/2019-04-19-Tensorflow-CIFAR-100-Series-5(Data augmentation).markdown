---
layout:     post
title:      Tensorflow CIFAR-100 Series (5)
author:     Omar Mohamed
tags: 		tensorflow ML CIFAR100
subtitle:  	Data augmentation
category:  Technical
---

In [part_4](https://omar-mohamed.github.io/technical/2019/04/19/Tensorflow-CIFAR-100-Series-4(Depthwise_conv)/) we trained our depth-wise convolution model.
But in this tutorial we will build upon [part_2](https://omar-mohamed.github.io/technical/2019/03/22/Tensorflow-CIFAR-100-Series-2(CNN)/) again because 
we only care about increasing accuracy. So we will be adding data augmentation to our normal CNN model.

## Data Augmentation

Like we discussed before, CIFAR-100 contains few images per class which makes training a model that generalies harder, so data augmentation can really be 
helpful to increase our generalizatoin. One of the best libraries out there for image augmentation is [imgaug](https://imgaug.readthedocs.io/en/latest/)
it takes a batch of images and begin to augment them as you specify. Here are some examples:

![image](https://user-images.githubusercontent.com/6074821/56441149-6783b700-62ec-11e9-8a0f-cfa904a1ce7a.png)

[Here](https://imgaug.readthedocs.io/en/latest/source/augmenters.html) you can find a list of all possible augmentations.

## Implementation

After installing the library and importing it, only a few lines will be added to our [part_2](https://omar-mohamed.github.io/technical/2019/03/22/Tensorflow-CIFAR-100-Series-2(CNN)/)
implementation. We will be augmenting the training batches randomly before they go into the model.

{% highlight python %}

seq = iaa.SomeOf((0, None), [
    iaa.Crop(px=(0, 12)),  # crop images from each side by 0 to 8px (randomly chosen)
    iaa.Fliplr(0.5),  # horizontally flip 50% of the images
    iaa.CoarseDropout((0.0, 0.20), size_percent=(0.02, 0.25), per_channel=0.5),
    # Drop 0 to 20% of all pixels by converting them to black pixels
], random_order=True)

{% endhighlight %}

Here we initialized our augmentation parameters. We will be using them randomly and in any combination. Cropping from the sides, flip left and right
,and dropping parts of the image. Of course you can add as many more as you want. 

And we will add one line in our training function to augment the batch:

{% highlight python %}

    # method to train one epoch of train data
    def train_epoch(dataset, labels, batch_size):
        data_size = dataset.shape[0]
        global step
        for offset in range(0, data_size, batch_size):
            batch_data = dataset[offset:(offset + batch_size), :]
			
			# notice this line
            batch_data = seq.augment_images(batch_data)

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

## Results

Training set accuracy:

![image](https://user-images.githubusercontent.com/6074821/54309347-9d8c9780-45d8-11e9-95e1-fe7869db44eb.png)
![image](https://user-images.githubusercontent.com/6074821/54309060-0fb0ac80-45d8-11e9-9b26-a15c67a27589.png)

Test set accuracy:

![image](https://user-images.githubusercontent.com/6074821/54309626-2dcadc80-45d9-11e9-9431-9d1933705325.png)
![image](https://user-images.githubusercontent.com/6074821/54309563-14299500-45d9-11e9-9cc2-eeb0414f09f8.png)

Final training set accuracy: 97.5% <br/>
Final training set loss: 0.1332 <br/>
Final test set accuracy: 65.3% <br/>
Final test set loss: 1.3090 <br/>

So a simple idea like data augmentation can really increase our accuracy by a large margin like you see here.



If you want to check the full state of the project until now click [here](https://github.com/omar-mohamed/Object-Classification-CIFAR-100) to go the repository. <br/>
See you in part 6.

