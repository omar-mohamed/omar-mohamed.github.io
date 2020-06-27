---
layout:     post
title:      Ultimate Image Classification
author:     Omar Mohamed
tags: 		tensorflow ML classification CNN finetuning
subtitle:  	Scene recognition walkthrough
category:  Technical
---

This is a walkthrough on how to use [ultimate_image_classification](https://github.com/omar-mohamed/ultimate-image-classification) repo
from start to finish. This repo is a general image classification module made using Tensorflow 2 and Keras
 that contains multiple pre-trained models you can use on any data. 

![ultimate image](https://user-images.githubusercontent.com/6074821/85757853-1a7d4b00-b710-11ea-8da6-256a97d6edd3.PNG)


## Problem

The problem we will try to tackle is scene recognition on a small dataset (around 450 images). 
There are 3 classes in the dataset which are (pool, operating room, gym).

## Walkthrough

- Download or clone the [repo](https://github.com/omar-mohamed/ultimate-image-classification)
- Install the requirements using pip in your python3.6+ environment. (Note you can switch to cpu version of tensorflow if you don't have a gpu)
- Download the dataset from [here] (https://drive.google.com/drive/folders/1JaXOQW6MUSE5aSFNv_nQyz45h_WtkoJy?usp=sharing)
- Extract the images inside 'data' folder in the repo to have 'data/images'
- To use the repo we need to have a csv to describe te data in this format:
![image](https://user-images.githubusercontent.com/6074821/85778023-705aee80-b722-11ea-936a-38b6d20329f8.png)

But since the data is already formatted to have a folder for each class with its images inside, we run make_csv_from_folders.py like this: <br/>
{% highlight python %}
python make_csv_from_folders.py --folder_path ./data/images --write_path ./data/all_data.csv
{% endhighlight %}

The script takes two parameters, the path to the folder, and the write path for the csv.

If it ran successfully, you will have a csv called 'all_data' inside data folder.

- Now we have a csv to describe the data, but we need to split it into train and test sets. We will do that by running split_train_test.py

{% highlight python %}
python split_train_test.py --csv_path ./data/all_data.csv --test_split_fraction 0.2 --shuffle True
{% endhighlight %}

The script takes three parameters: the path to the csv, the test set split fraction [0,1] (it will automatically calculate the same percentage on all classes), and an option to shuffle the data.
The script will generate training_set.csv and testing_set.csv next to all_data.csv.

- Now everything is ready to actually start training. We will open 'configs.py' in which we can control all aspects of our training:

{% highlight python %}
self.define('train_csv', './data/training_set.csv',
            'path to training csv containing the images names and the labels')
self.define('test_csv', './data/testing_set.csv',
            'path to testing csv containing the images names and the labels')
self.define('image_directory', '',
            'this path will be concatenated in front of the path in the csv. If the path in the csv is already complete leave it empty')
self.define('visual_model_name', 'DenseNet121',
            'select from (VGG16, VGG19, DenseNet121, DenseNet169, DenseNet201, Xception, ResNet50, ResNet101, ResNet152, ResNet50V2, ResNet101V2, ResNet152V2, InceptionV3, InceptionResNetV2, NASNetMobile, NASNetLarge, MobileNet, MobileNetV2, EfficientNetB0 to EfficientNetB7). Note that the classifier layer is removed by default.')
self.define('image_target_size', (224, 224, 3), 'the target size to resize the image')
self.define('num_epochs', 100, 'maximum number of epochs')
self.define('csv_label_column', 'class', 'the name of the label column in the csv')
self.define('classes', self.get_classes_list(self.train_csv, self.csv_label_column),
            'the names of the output classes. It will get this automatically from the csv.')
self.define('multi_label_classification', False,
            'determines if this is a multi label classification problem or not. It changes the loss function and the final layer activation from softmax to sigmoid')
self.define('classifier_layer_sizes', [0.4],
            'a list describing the hidden layers of the classifier. Example [10,0.4,5] will create a hidden layer with size 10 then dropout wth drop prob 0.4, then hidden layer with size 5. If empty it will connect to output nodes directly after flatten.')
self.define('conv_layers_to_train', -1,
            'the number of layers that should be trained in the visual model counting from the end side. -1 means train all and 0 means freezing the visual model')
self.define('use_imagenet_weights', True, 'initialize the visual model with pretrained weights on imagenet')
self.define('pop_conv_layers', 0,
            'number of layers to be popped from the visual model. Note that the imagenet classifier is removed by default so you should not take them into considaration')
self.define('final_layer_pooling', 'avg', 'the pooling to be used as a final layer to the visual model')
self.define('load_model_path', '',
            'a path containing the checkpoints. If provided the system will continue the training from that point or use it in testing.')
self.define('save_model_path', 'saved_model',
            'where to save the checkpoints. The path will be created if it does not exist. The system saves every epoch by default')
self.define('save_best_model_only', True,
            'Only save the best weights according to validation accuracy or auroc')
self.define('learning_rate', 1e-3, 'The optimizer learning rate')
self.define('learning_rate_decay_factor', 0.1,
            'Learning rate decay factor when validation loss stops decreasing')
self.define('reduce_lr_patience', 3,
            'The number of epochs to reduce the learning rate when validation loss is not decreasing')
self.define('minimum_learning_rate', 1e-7, 'The minimum possible learning rate when decaying')

self.define('optimizer_type', 'Adam', 'Choose from (Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam)')
self.define('gpu_percentage', 0.95, 'gpu utilization. If 0 it will use the cpu')
self.define('batch_size', 16, 'batch size for training and testing')
self.define('multilabel_threshold_range', [0.01, 0.99],
            'The threshold from which to detect a class. Only used with multi label classification. It will automatically search for the best threshold in the range and choose it ')
self.define('generator_workers', 4, 'The number of cpu workers preparing batches for the GPU.')
self.define('generator_queue_length', 12, 'The maximum number of batches in the queue to be trained on.')
self.define('show_model_summary', True, 'A flag to show or hide the model summary')
self.define('positive_weights_multiply', 1.0,
            'Controls the class_weight ratio. Higher value means higher weighting of positive samples. Only works if use_class_balancing is set to true')
self.define('use_class_balancing', True,
            'If set to true it will automatically balance the classes by settings class weights')
self.define('cnn_downscaling_factor', 0,
            'Controls the cnn layers responsible for downscaling the input image. if input image is 512x512 and downscaling factor is set to 2 then the downscaling cnn will output image with size 128x128. Note it is a learnable net and if set to 0 it will skip it')
self.define('cnn_downscaling_filters', 64, 'Number of filters in the downscaling model')
{% endhighlight %}

This is the most important file in the repo and contains varied options to control. Most 
importantly, we need to link the training and testing csvs as seen in this part:

{% highlight python %}
self.define('train_csv', './data/training_set.csv',
            'path to training csv containing the images names and the labels')
self.define('test_csv', './data/testing_set.csv',
            'path to testing csv containing the images names and the labels')
{% endhighlight %}

You can definitely try to tweak other parameters like changing the base model, the batch size, or learning rate. If you have memory problems 
try reducing the batch size or choose a smaller model.

- Now we can start the training: 
{% highlight python %}
python train.py
{% endhighlight %}

The training will automatically save the best checkpoint to the folder specified in configs. In our case:
{% highlight python %}
self.define('save_model_path', 'saved_model',
            'where to save the checkpoints. The path will be created if it does not exist. The system saves every epoch by default')
{% endhighlight %}

So after training for some time, we will find a model in 'saved_model' folder. You will also find the tensorboard logs, the configs you used 
, and a training log csv that keeps track of loss and accuracies over the training.

- Now that we have a model we can actually test it to get the full metrics and evaluation. First we need to specify a load model path in configs:
{% highlight python %}
        self.define('load_model_path', '',
                    'a path containing the checkpoints. If provided the system will continue the training from that point or use it in testing.')
{% endhighlight %}

Note that if we run train.py again now it will continue training from this checkpoint, but to test we will run the test script:

{% highlight python %}
python test.py
{% endhighlight %}

The test script will provide you with many metrics to judge the system like:

![image](https://user-images.githubusercontent.com/6074821/85793670-a48bda80-b735-11ea-9fb5-10a4cde986b0.png)

- Now we can also draw our activation maps by using [Grad-Cam](https://arxiv.org/abs/1610.02391). We do that 
by running draw_activations script:
{% highlight python %}
python draw_activations.py
{% endhighlight %}

The script loads the model specified in 'load_model_path' in configs and saves the results next to the model 
, in 'save_model_path' in configs, inside a folder called cam_output. The results will be similar to these:

<br/>
 <img src="https://user-images.githubusercontent.com/6074821/85793918-0c422580-b736-11ea-85ac-3b7bfc883593.png" width="300" height="250" align="left">
 <img src="https://user-images.githubusercontent.com/6074821/85794105-5aefbf80-b736-11ea-8257-0abc79bc2e02.png" width="300" height = "250" align="left">
 <img src="https://user-images.githubusercontent.com/6074821/85794298-a6a26900-b736-11ea-9a5d-752d81a479fa.png" width="300" height = "250" align="left|top">

 
- Although there is a full tensorboard support during training, you might want to draw a simple figure of the 
training and test accuracies through the epochs. To do that you can run plot_training_log script:
{% highlight python %}
python plot_training_log.py --train_log_path ./saved_model/training_log.csv
{% endhighlight %}

It takes the path to the csv, if not specified it will look for it inside 'save_model_path' in configs.
The results will be like:
![image](https://user-images.githubusercontent.com/6074821/85786603-b5831e80-b72a-11ea-9769-117b2ccd44a2.png)


- After being satisfied with a model, we can further compress it to a smaller size since the normal saving 
of the model saves information related to the training operation. We can compress the model by calling compress_weights script:
{% highlight python %}
python compress_weights --model_path ./saved_model/best_model.hdf5
{% endhighlight %}

If model path is not provided it will automatically take the path specified in 'load_model_path' in configs. 
The script will then output two files 'best_model_compressed.h5' and 'best_model_compressed.json' in the same 
directory of the model. This new compressed version can be loaded using 'custom_load_model' method in utils.py:

{% highlight python %}
def custom_load_model(load_path, model_name):
    path = os.path.join(load_path, model_name)
    # load json and create model
    json_file = open('{}.json'.format(path), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # # load weights into new model
    loaded_model.load_weights("{}.h5".format(path))
    print("Loaded model from disk")
    return loaded_model
{% endhighlight %}

The compressed model will be faster to load in inference mode.

### Multi-label classification

The repo also supports multi-label classification, but of course there are some differences like:

- The data csv should have the labels separated by '<span>\$</span>' if an image has more than one class associated with it.
 For example: class1<span>\$</span>class2. In this case you should provide the csv for the training and test sets.
 
- In multi-label classification mode, the loss function is automatically switched to binary cross entropy, instead of categorical cross entropy, and also
the final activations will be set to sigmoid instead of softmax.
 
 - The metrics for measuring the performance are different in multi-label classification, mainly we use 
 AUC-ROC to determine how well the model is performing and save the best model accordingly.
 So in the 'save_model_path' you will have extra files like '.training_stats.json' and 'best_auroc.log' to describe the training so far. 
 And of course running the test script will provide a different output. There is also an exact match accuracy, the 
 repo automatically searches for the best threshold, and output it, from the range specified in the configs in 'multilabel_threshold_range' field.
 
 - When running the draw_activations script in multi-label you will only get activations for the highest class
 predicted, so you might want to tweak the strategy of which classes (top k for instances, or with a threshold)
 to draw the activations according to.
 
 - You can run the same example as a multi-label classification problem by setting this flag in configs:
 {% highlight python %}
self.define('multi_label_classification', True,
            'determines if this is a multi label classification problem or not. It changes the loss function and the final layer activation from softmax to sigmoid')
{% endhighlight %}

When running the test script you will get results similar to these:
![image](https://user-images.githubusercontent.com/6074821/85924018-be413500-b88f-11ea-9ab5-b9e8a9781df5.png)


### Data augmentations

By default we use the following augmentations:
- Rotations
- Scaling
- Flipping
- Color augmentations 
- Contrast augmentations
- Shear
- Translation

In case you want to add or remove some augmentations, you can tweak 'augmenter.py'. We use imgaug library for
the augmentations, so you can visit this [page](https://imgaug.readthedocs.io/en/latest/source/overview_of_augmenters.html) 
for more possible augmentations.

![image](https://user-images.githubusercontent.com/6074821/85763025-9a0d1900-b714-11ea-8d32-553eff595106.png)

### Thank you
If you have any questions, please leave a comment here or add an issue to the repo.
