# Semantic Segmentation of 2D images using a U-Net

by : Jhonatan Contreras 

### This repository, provides model definition, helper functions.  
  
It is a deep convolutional neural network learning system for semantic image segmentation  
with the following features:  
1.  The architecture goes through a series of down-sampling layers  
that reduce the dimensionality along the spatial dimension.  
Pretrain down-sampling net ==> DenseNet121  
2.  This is followed by a series of up-sampling layers,  
it increases the dimensionality along the spatial dimensions,  
Pretrain up-sampling net ==> pix2pix  
3.  The output layer has the same spatial dimensions as the original input image  
  
###  See the following papers for more details:  
 Badrinarayanan, V., Kendall, A., & Cipolla, R. (2017). Segnet: A deep convolutional encoder-decoder architecture for image segmentation. IEEE transactions on pattern analysis and machine intelligence, 39(12), 2481-2495.  
 
 Isola, P., Zhu, J. Y., Zhou, T., & Efros, A. A. (2017). Image-to-image translation with conditional adversarial networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1125-1134).

### Training in Google Colab (https://colab.research.google.com/)

- Copy file: "sem_seg.ipynb"
- Download and copy dataset (cityscapes) in Google Drive
- Run file with GPU option
- Download "my_h5model.h5" file 
- Run "test_live.h5" for inference using a webcam

