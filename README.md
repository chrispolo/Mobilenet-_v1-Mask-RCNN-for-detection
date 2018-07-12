MobileNet version of mask rcnn - can be extended to mobile devices

1, the installation environment:  
  
1, computer environment:  
  
Mask R-CNN is based on Python3, Keras, TensorFlow.  
  
- Python 3.4+  
- TensorFlow 1.3+  
- Keras 2.0.8+  
- Jupyter Notebook  
- Numpy, skimage, scipy, Pillow, cython, h5py  
- opencv 2.0  
  
2. Pre-weight download: [mobile_mask_coco.h5 download] (https://download.csdn.net/download/ghw15221836342/10536729)  
  
3. If you need to train or test on the COCO dataset, you need to install `pycocotools`, `clone` down, `make` to generate the corresponding file, and copy the generated pycocotools folder into the project folder after make:  
  
Linux: [https://github.com/waleedka/coco](https://github.com/waleedka/coco)  
  
Windows: [https://github.com/philferriere/cocoapi](https://github.com/philferriere/cocoapi). You must have the Visual C++ 2015 build tools on your path (see the repo for additional details)  
  
4. [MS COCO Dataset] (https://translate.googleusercontent.com/translate_c?depth=1&hl=en-US&prev=search&rurl=translate.google.com.hk&sl=en&sp=nmt4&u=http://cocodataset.org/&xid=17259,15700021,15700124,15700149,15700168,15700173,15700186,15700201&usg=ALkJrhja-wzxrDmjJ1K1d2ySakONgKI2Rw#home)(Ubuntu recommends using the wget command to download directly from Ubuntu terminal) I use the coco2014 dataset for training, including more than 80 categories .
  

## 2.Training process:

If you need to train yourself, open the terminal directly in the project.:

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


MobileNet_v1+FPN feature extraction for the backbone network to achieve mobilenet version of the target detection. I replaced the backbone network with mobilenet for the cocodataset 2014 dataset, 80k training set and 35k validation set training, before communicating at Shanghai University The cloud platform created the container, but the model trained in the NvidiaTesla P100 has no way to target detection.

On the communication cloud platform, a docker with a deep learning framework and a corresponding environment is created. The NvidiaTesla M10 8G graphics card is used for training, using 1 GPU training (such an effective batch size is 16) 160k iterations, learning rate At 0.001, the learning rate is divided by 10 at 120k iterations. At the same time, the number of training steps is 160 steps, each step is iterated 1000 times, and the weight of each step training is saved. The total training time is 72 hours.
![](https://img-blog.csdn.net/20180712113709280?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2dodzE1MjIxODM2MzQy/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)  
Figure 1 Loss changes in the MobileNet training process shown by Tensorboard

  

![](https://img-blog.csdn.net/20180712113746301?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2dodzE1MjIxODM2MzQy/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

  
Figure 2 Changes in the loss of the MobileNet training process validation set demonstrated by Tensorboard

  

![](https://img-blog.csdn.net/20180712113822988?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2dodzE1MjIxODM2MzQy/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)  

Figure 3: Change in loss value during iteration during training

## 2.MobileNet_v1 principle


MobileNet-v1 is a lightweight deep neural network proposed by Google in April 2017 for embedded devices such as mobile phones for mobile and embedded vision applications. MobileNet-v1 is based on a streamlined architecture that uses deep separable convolution to build a lightweight deep neural network, introducing two simple global hyperparameters that effectively balance between latency and accuracy. These two hyperparameters allow the model builder to choose the appropriate size model for its application based on the constraints of the problem.

  

Convolutional neural networks have become more popular since the Alex Net model won the ImageNet Championship in 2012. In the face of massive data, in order to improve the accuracy of testing, people often choose to increase the accuracy by increasing the number of layers of the network and more complex neural networks. However, this method requires higher computing power of the device, and the trained model has a larger memory, and the requirements for the experimental device are also high during operation, which is not conducive to the application and popularization of related technologies. For example, some embedded platforms require a model with smaller memory, faster testing, and flexibility due to hardware resources. The key to MobileNet-v1 is to decompose the convolution kernel and decompose the standard convolution kernel into a deep convolution and a 1*1 convolution. Deep convolution applies the corresponding convolution kernel to each channel, and the 1*1 convolution kernel is used to combine the channel outputs. The following figure shows the decomposition process of the standard convolutional layer. By this decomposition, the amount of calculation can be reduced and the test speed can be improved.

![](https://img-blog.csdn.net/20180712114217994?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2dodzE1MjIxODM2MzQy/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)  

Figure (a) 3D convolution (b) channel-by-channel 2D convolution and (c) 3D 1*1 convolution

  

	The traditional convolution shown in Figure (a) is convoluted by using the same dimensions as the input and output. The calculated amount is T1=DK· DK · M · N · DF · DF, M, N are input and The number of channels output. Figure (b) uses a Depthwise convolution kernel. First, a set of convolution kernels with a channel number of one is used. The number of two-dimensional convolution kernels is the same as the number of input channels. The feature map is obtained after processing by channel-by-channel convolution, and the 1*1 convolution kernel is selected to process the output feature map. The calculation amount of the channel-by-channel 2D convolution kernel is: T2=DK· DK · M · N · DF · DF, the calculation amount of the 3*1 convolution kernel of 3D is: T3=DK· DK · M · DF · DF. Therefore, the calculation amount of this combination is: T2+T3. The deep-wise way of convolution is compared to the conventional 3D convolution calculation: (T2+T3)/T1=1/N +1/DK2. The network structure shown in the figure below is a traditional standard convolution network and a deep-wise convolution network structure.

:![](https://img-blog.csdn.net/201807121143253?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2dodzE1MjIxODM2MzQy/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)


Traditional 3D convolution (left) and deep-wise convolutional network structure (right)

As can be seen from the figure, the deep-wise convolution and the subsequent 1x1 convolution are relatively independent. The 1*1 convolution drop channel is used to reduce the amount of calculation. Deep-wise combined with the 1x1 convolution method instead of the traditional convolution is not only theoretically more efficient, but because of the large number of 1x1 convolutions, this can be done directly using a highly optimized math library, 95% in MobileNet. The calculated amount and 75% of the parameters belong to the 1x1 convolution.

  

## 3.Comparative Results
	
	  
![](https://img-blog.csdn.net/20180712114912552?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2dodzE1MjIxODM2MzQy/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

Figure 1 shows the result of the mask rcnn test.

![](https://img-blog.csdn.net/20180712115025478?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2dodzE1MjIxODM2MzQy/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)  

Figure 2 shows the test results of Mobilenet v1.

![](https://img-blog.csdn.net/20180712115114404?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2dodzE1MjIxODM2MzQy/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)  
Figure 3 shows the result of the mask rcnn test.

![](https://img-blog.csdn.net/20180712115147636?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2dodzE1MjIxODM2MzQy/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)  

Figure 4 shows the test results of Mobilenet v1.

  

## 4. 模型性能分析

准确率(Precision):识别过程中,正确识别的图片占图片总数的比值,召回率(Recall):正确识别个数,占测试集中该类总数的比值,平均准确率(AveragePrecison):AP,不同类别准确率的平均值.平均召回率(AverageRecall):AR,不同类别召回率的平均值。

对Resnet50和MobileNet为主干网络的训练的模型,在coco验证集上,随机选取500张图片进行检验,结果如下图所示。

![](https://img-blog.csdn.net/20180712115528311?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2dodzE1MjIxODM2MzQy/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)  

Figure 1 Results of Resnet AP and AR on 500 images

![](https://img-blog.csdn.net/20180712115557597?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2dodzE1MjIxODM2MzQy/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)  

Figure 2 Results of MobileNet AP and AR on 500 pictures


Through the test of 500 pictures, the features extracted by Resnet50 as the backbone network can be obtained. The average accuracy and average recall rate are higher than those of MobileNet_v1 under different Iou. This can also be compared with the test results shown above. It can be seen that for the instance segmentation of the target, the Resnet50 segmentation detection is more accurate. The MobleNet version of MaskR-CNN overlaps the overlapping target detection and the accuracy is not high enough. However, the classification of objects and the return of the border are close.

The test speeds of the two models are also different, and 500 pictures are selected on the TeslaM10 graphics card for testing.

The total detection time of Resnet50 is 511s, and the average detection time is 1.02s. The total detection time of MobileNet is 370s, and the average detection time is 0.74s. It can be seen that the model of MobileNet training is more flexible and the detection efficiency is higher.

ResNet50 trained model memory is 245M, MobileNetV1 trained model memory is 93M, which reflects the MobileNet model is small, easy to transplant to mobile devices. Memory size, as shown below:

![](https://img-blog.csdn.net/20180712115625219?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2dodzE1MjIxODM2MzQy/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)  

Figure 3 ResNet 50 training model memory size

![](https://img-blog.csdn.net/20180712115645370?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2dodzE1MjIxODM2MzQy/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)  

Figure 4 MobileNet training model size
