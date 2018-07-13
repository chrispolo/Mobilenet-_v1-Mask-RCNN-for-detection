# MobileNet_v1 -Mask RCNN for detection

MobileNet version of mask rcnn - can be extended to mobile devices

# 1.Installation	 enviroment

1, computer environment:

Mask R-CNN is based on Python3, Keras, TensorFlow.

-   Python 3.4+
-   TensorFlow 1.3+
-   Keras 2.0.8+
-   Jupyter Notebook
-   Numpy, skimage, scipy, Pillow, cython, h5py
-   opencv 2.0

2.  Pre-weight download: [mobile_mask_coco.h5 download] ([https://download.csdn.net/download/ghw15221836342/10536729](https://download.csdn.net/download/ghw15221836342/10536729)),if you don't have the account,please send email to me（guohongwu928566@gmail.com）!
    
3.  If you need to train or test on the COCO dataset, you need to install  `pycocotools`,  `clone`  down,  `make`  to generate the corresponding file, and copy the generated pycocotools folder into the project folder after make:
    

Linux:  [https://github.com/waleedka/coco](https://github.com/waleedka/coco)

Windows:  [https://github.com/philferriere/cocoapi](https://github.com/philferriere/cocoapi). You must have the Visual C++ 2015 build tools on your path (see the repo for additional details)

4.  [MS COCO Dataset](Ubuntu)recommends using the wget command to download directly from Ubuntu terminal) I use the coco2014 dataset for training, including more than 80 categories .

## 2.Training process
(I use the Nvidia Tesla M10 8g gpu)
If you need to train yourself, open the terminal directly in the project.:

```
# Train a new model starting from pre-trained COCO weights
python3 coco.py train --dataset=/path/to/coco/ --model=coco

# Train a new model starting from ImageNet weights. Also auto download COCO dataset
python3 coco.py train --dataset=/path/to/coco/ --model=imagenet --download=True

# Continue training a model that you had trained earlier
python3 coco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

# Continue training the last model you trained
python3 coco.py train --dataset=/path/to/coco/ --model=last

# Run COCO evaluatoin on the last model you trained
python3 coco.py evaluate --dataset=/path/to/coco/ --model=last

```

MobileNet_v1+FPN feature extraction for the backbone network to achieve mobilenet version of the target detection. I replaced the backbone network with mobilenet for the cocodataset 2014 dataset, 80k training set and 35k validation set training, before communicating at Shanghai University The cloud platform created the container, but the model trained in the NvidiaTesla P100 has no way to target detection.

On the communication cloud platform, a docker with a deep learning framework and a corresponding environment is created. The NvidiaTesla M10 8G graphics card is used for training, using 1 GPU training (such an effective batch size is 16) 160k iterations, learning rate At 0.001, the learning rate is divided by 10 at 120k iterations. At the same time, the number of training steps is 160 steps, each step is iterated 1000 times, and the weight of each step training is saved. The total training time is 72 hours.

![1](https://github.com/chrispolo/Mobilenet-_v1-Mask-RCNN-for-detection/blob/master/project%20photo/1.jpg)


                                                                                  Figure 1 Loss changes in the MobileNet training process shown by Tensorboard

![Figure 2 Changes in the loss of the MobileNet training process validation set demonstrated by Tensorboard](https://github.com/chrispolo/Mobilenet-_v1-Mask-RCNN-for-detection/blob/master/project%20photo/2.png)

                                                                                   Figure 2 Changes in the loss of the MobileNet training process validation set demonstrated by Tensorboard

![Figure 3: Change in loss value during iteration during training](https://github.com/chrispolo/Mobilenet-_v1-Mask-RCNN-for-detection/blob/master/project%20photo/3.png)

             Figure 3: Change in loss value during iteration during training

## 3. MobileNet_v1 principle

MobileNet-v1 is a lightweight deep neural network proposed by Google in April 2017 for embedded devices such as mobile phones for mobile and embedded vision applications. MobileNet-v1 is based on a streamlined architecture that uses deep separable convolution to build a lightweight deep neural network, introducing two simple global hyperparameters that effectively balance between latency and accuracy. These two hyperparameters allow the model builder to choose the appropriate size model for its application based on the constraints of the problem.

Convolutional neural networks have become more popular since the Alex Net model won the ImageNet Championship in 2012. In the face of massive data, in order to improve the accuracy of testing, people often choose to increase the accuracy by increasing the number of layers of the network and more complex neural networks. However, this method requires higher computing power of the device, and the trained model has a larger memory, and the requirements for the experimental device are also high during operation, which is not conducive to the application and popularization of related technologies. For example, some embedded platforms require a model with smaller memory, faster testing, and flexibility due to hardware resources. The key to MobileNet-v1 is to decompose the convolution kernel and decompose the standard convolution kernel into a deep convolution and a 11 convolution. Deep convolution applies the corresponding convolution kernel to each channel, and the 11 convolution kernel is used to combine the channel outputs. The following figure shows the decomposition process of the standard convolutional layer. By this decomposition, the amount of calculation can be reduced and the test speed can be improved.


## 4.Comparative Results
![Figure 1 shows the result of the mask rcnn test.](https://github.com/chrispolo/Mobilenet-_v1-Mask-RCNN-for-detection/blob/master/project%20photo/4.png)
 
                Figure 1 shows the result of the mask rcnn test

![Figure 2 shows the test results of Mobilenet v1.](https://github.com/chrispolo/Mobilenet-_v1-Mask-RCNN-for-detection/blob/master/project%20photo/5.png)
  
                 Figure 2 shows the test results of Mobilenet v1

![Figure 3 shows the result of the mask rcnn test.](https://github.com/chrispolo/Mobilenet-_v1-Mask-RCNN-for-detection/blob/master/project%20photo/6.png)

                 Figure 3 shows the result of the mask rcnn test.

![Figure 4 shows the test results of Mobilenet v1.](https://github.com/chrispolo/Mobilenet-_v1-Mask-RCNN-for-detection/blob/master/project%20photo/7.png)
 
                  Figure 4 shows the test results of Mobilenet v1.

## 5.Model performance analysis
Precision: The ratio of correctly recognized pictures to the total number of pictures during the recognition process. Recall: correctly identify the number, the ratio of the total number of the test sets, the average accuracy (AveragePrecison): AP, different Average of category accuracy. Average recall (AverageRecall): AR, the average of the recall rates for different categories.

For the training model of Resnet50 and MobileNet backbone network, 500 pictures were randomly selected for testing on the coco verification set. The results are shown in the figure below.

![Figure 1 Results of Resnet AP and AR on 500 images](https://github.com/chrispolo/Mobilenet-_v1-Mask-RCNN-for-detection/blob/master/project%20photo/8.png)

                         Figure 1 Results of Resnet AP and AR on 500 images

![Figure 2 Results of MobileNet AP and AR on 500 pictures](https://github.com/chrispolo/Mobilenet-_v1-Mask-RCNN-for-detection/blob/master/project%20photo/9.png)

                     Figure 2 Results of MobileNet AP and AR on 500 pictures

Through the test of 500 pictures, the features extracted by Resnet50 as the backbone network can be obtained. The average accuracy and average recall rate are higher than those of MobileNet_v1 under different Iou. This can also be compared with the test results shown above. It can be seen that for the instance segmentation of the target, the Resnet50 segmentation detection is more accurate. The MobleNet version of MaskR-CNN overlaps the overlapping target detection and the accuracy is not high enough. However, the classification of objects and the return of the border are close.

The test speeds of the two models are also different, and 500 pictures are selected on the TeslaM10 graphics card for testing.

The total detection time of Resnet50 is 511s, and the average detection time is 1.02s. The total detection time of MobileNet is 370s, and the average detection time is 0.74s. It can be seen that the model of MobileNet training is more flexible and the detection efficiency is higher.

ResNet50 trained model memory is 245M, MobileNetV1 trained model memory is 93M, which reflects the MobileNet model is small, easy to transplant to mobile devices. Memory size, as shown below:

![  Figure 3 ResNet 50 training model memory size](https://github.com/chrispolo/Mobilenet-_v1-Mask-RCNN-for-detection/blob/master/project%20photo/10.png)

                             Figure 3 ResNet 50 training model memory size

![Figure 4 MobileNet training model size](https://github.com/chrispolo/Mobilenet-_v1-Mask-RCNN-for-detection/blob/master/project%20photo/11.png)

                              Figure 4 MobileNet training model size
