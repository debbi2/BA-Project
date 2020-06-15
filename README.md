# Visualizing Deep Learning-based Brain Tumor Classification Model Using CAM

Deep learning using convolutional neural network brings huge success in image classification and segmentation. Furthermore, the appearance of class activation mapping (CAM) makes it possible to understand deep learning image processing principles. Therefore, a numerous medical image research uses this model to see how well the model is optimized on medical image data. However, in most cases, CAM is only used for qualitative analysis of the model. In this project, we did a quantitative analysis for CAM to measure how well the deep learning model can detect the tumor and if it classifies its type correctly. To conduct, deep learning model for analyzing brain MRI images for brain tumor classification is created. This model is then visualized using CAM in the form of heatmaps. We convert this heatmap into pixel-wise mask which is compared with the ground truth brain tumor mask image using intersection-over-union (IoU) as a evaluation metric. As a result, even though the model accuracy is higher than 0.98, the average IoU is 0.153. In other words, the CAM cannot segment very well. Also, in quantitative analysis, extreme cases of which the model could not correctly highlight tumor area although, tumor can be easily distinguished on human eyes manually. As a result, classification model cannot be fully trusted and be careful using the model.

## Experimental Environment
All experiments are conducted using Python in the following hardware environment:
* GeForce GTX 1080 8GB
* Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz / RAM 16Gb

The software used in the experiment are:
* Python version 3.7.6
* Conda version 4.8.3
* CUDA version 10.2
* Scikit-learn version 0.22.1
* Numpy version 1.18.1
* Torch version 1.4.0
* Torchvision version 0.5.0

## Dataset
Dataset used in this project is available [here](https://figshare.com/articles/brain_tumor_dataset/1512427). The dataset consists of 3,064 brain MRI images from 233 patients with three different types of brain tumors: Meningioma (708 slices), Glioma (1,426 slices), and Pituitary tumour (930 slices). The dataset is split into five files with names: "dataset0.csv", "dataset1.csv", "dataset2.csv", "dataset3.csv", "dataset4.csv" 

![data image](D:/BA4/Research/BA_project/dataset.jpg)
<img src = "D:/BA4/Research/BA project/dataset.jpg">

## Model
We have created three models using pre-trained model of AlexNet, VGG16, and VGG19. Global average pooling layer and classifier layer is modified. 5-fold cross validation is conducted for every models. VGG19 is the best performing model compared to AlexNet and VGG16. The VGG19 model trained with “dataset0.csv”, “dataset1.csv”, “dataset2.csv”, and “dataset3.csv” shows the highest accuracy of 0.98 and is used to visualize using CAM.

## Visualization
A heatmap is created by getting the dot product of a feature map of the final convolutional layer and weights of GAP layer. The heatmap indicates a discriminative region for a class of interest. It is then binarized at a threshold of 95th percentile to create a segmentation mask image.

![result image1](D:\BA4\Research\BA_project\result1.jpg)
![result image2](D:\BA4\Research\BA_project\result2.jpg)

## IoU
Segmentation mask images of the test set is compared with the ground truth mask images for tumors are compared using IoU. The mean IoU (mIoU) of 0.153 is obtained. This indicates there are only 15.3 %  similarity of mask and ground truth images.
Even though the classification model can well distinguish brain tumors with accuracy of 0.98, the model cannot detect a tumor precisely in its exact shape and size, and sometimes detect wrong area (i.e. corners of the image) as a tumor.