---
layout:     post
title:      Tensorflow CIFAR-100 Series (1)
author:     Omar Mohamed
tags: 		tensorflow ML CIFAR100 preprocessing
subtitle:  	CIFAR-100 preprocessing and saving
category:  Technical
---

In this series, I will be discussing some training models and techniques on the CIFAR100 dataset using tensorflow. Since this is the first part, we will put some ground work, like downloading, extracting, preprocessing, and saving the dataset for training. In the next tutorial, we will begin the training using a simple CNN. And as the series goes on, we will investigate some modifications like (DropBlocks, Resnets, Depthwise convolutions, Self attention)

## Environment Used:
- Python 3.6.1
- Tensorflow 1.10
- imgaug 0.2.8
- opencv-python 4.0.0

## Problem formulation and dataset info

Given a blurry image, the task is to classify it into one of the 100 classes in CIFAR-100.
The dataset consists of 60000 32x32 colour images in 100 classes, with 600 images per class. There are 50000 training images and 10000 test images. 

Link: [CIFAR100_Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

![plt](https://user-images.githubusercontent.com/6074821/52181190-11789a80-27f8-11e9-8104-7751bfce2e18.png)


## Download and extract the dataset

This script will begin to download the CIFAR100 dataset in the project folder

{% highlight python %}

import os
import sys
from six.moves.urllib.request import urlretrieve


url = 'https://www.cs.toronto.edu/~kriz/'
last_percent_reported = None
data_root = '.'  # Change me to store data elsewhere



def download_progress_hook(count, blockSize, totalSize):
    """A hook to report the progress of a download. This is mostly intended for users with
    slow internet connections. Reports every 5% change in download progress.
    """
    global last_percent_reported
    percent = int(count * blockSize * 100 / totalSize)

    if last_percent_reported != percent:
        if percent % 5 == 0:
            sys.stdout.write("%s%%" % percent)
            sys.stdout.flush()
        else:
            sys.stdout.write(".")
            sys.stdout.flush()

        last_percent_reported = percent


def maybe_download(filename, expected_bytes, force=False):
    """Download a file if not present, and make sure it's the right size."""
    dest_filename = os.path.join(data_root, filename)
    if force or not os.path.exists(dest_filename):
        print('Attempting to download:', filename)
        filename, _ = urlretrieve(url + filename, dest_filename, reporthook=download_progress_hook)
        print('\nDownload Complete!')
    statinfo = os.stat(dest_filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', dest_filename)
    else:
        raise Exception(
            'Failed to verify ' + dest_filename + '. Can you get to it with a browser?')
    return dest_filename


maybe_download('cifar-100-python.tar.gz', 169001437)

{% endhighlight %}

If the download is successful, run the following script to extract the dataset

{% highlight python %}
import os
import sys
import tarfile

data_root = '.'

def maybe_extract(filename, force=False):
    root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
    if os.path.isdir(root) and not force:
        # You may override by setting force=True.
        print('%s already present - Skipping extraction of %s.' % (root, filename))
    else:
        print('Extracting data for %s. This may take a while. Please wait.' % root)
        tar = tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall(data_root)
        tar.close()


dataset = os.path.join(data_root, 'cifar-100-python.tar.gz')

maybe_extract(dataset)
{% endhighlight %}

Now we are ready to start the dataset preprocessing

## Dataset preprocessing and saving

The first thing we should do is to load the dataset:

{% highlight python %}
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

# a method to load a pickle file
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


# load train, test, and meta
print("Loading Data..")

train_dec = unpickle('./cifar-100-python/train')
test_dec = unpickle('./cifar-100-python/test')
meta = unpickle('./cifar-100-python/meta')

# load data and labels
train_data = train_dec[b'data']
train_labels = np.array(train_dec[b'fine_labels'])
test_data = test_dec[b'data']
test_labels = np.array(test_dec[b'fine_labels'])
label_names = meta[b'fine_label_names']

{% endhighlight %}

Now we have the training and test sets loaded into their respective variables, and we also load the meta data which holds information like label index name.

Next we need to reformat the images and shuffle the training set

{% highlight python %}

# a method to return dataset in format [num_images,image_height,image_width,num_channels]
def reshape(dataset):
    return np.reshape(dataset, (-1, 3, 32, 32)).transpose((0, 2, 3, 1))


# a method to shuffle dataset and labels
def randomize(dataset, labels):
    permutation = np.random.permutation(dataset.shape[0])
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels
	
# reshape and shuffle
print("Shuffling Data..")

train_data = reshape(train_data)
test_data = reshape(test_data)
train_data, train_labels = randomize(train_data, train_labels)

{% endhighlight %}

If you take a look at the images in the dataset, you will notice how blurry they are. So let's try to combat that by applying an [unsharp_masking_kernel](https://en.wikipedia.org/wiki/Unsharp_masking) to sharpen the blurry images

![image](https://user-images.githubusercontent.com/6074821/53117205-0ee5a700-3553-11e9-969c-e5bc84c2299b.png)

{% highlight python %}
# a method to sharpen the blurry images
def sharpen(dataset):
    # img = rgb2gray(img)
    kernel = (-1 / 256.0) * np.array(
        np.asarray([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, -476, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]]))

    for i in range(dataset.shape[0]):
        dataset[i] = cv2.filter2D(dataset[i], -1, kernel)

    return dataset

train_data = sharpen(train_data)
test_data = sharpen(test_data)

{% endhighlight %}

Now I know what you are thinking... We should normalize the values to be between [-1, 1] or something similar. And you are right, but we will do it during the training to make data augmentation easier.
With that out of the way, the only thing we have left is to save the data.

{% highlight python %}
pickle_file = 'CIFAR_100_processed.pickle'

# pickle data after pre processing
try:
    f = open(pickle_file, 'wb')
    save = {
        'train_dataset': train_data,
        'train_labels': train_labels,
        'test_dataset': test_data,
        'test_labels': test_labels,
        'label_names': label_names
    }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
    print("Done")
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise

{% endhighlight %}

Now we should be ready for training, which will be done in the next part of the series. If you want to check the full state of the project until now click [here](https://github.com/omar-mohamed/Object-Classification-CIFAR-100) to go the repository. <br/>
See you in [part_2](https://omar-mohamed.github.io/technical/2019/03/22/Tensorflow-CIFAR-100-Series-2(CNN)/).
