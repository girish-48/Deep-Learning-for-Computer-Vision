# This folder contains the MPS and the project for the course CS444 : Deep Learning for Computer Vision at UIUC
# I had done the assignments and project by myself, this repo is to keep track of it 
---

The five MPs are listed as follows : 
* Simple Linear Classifiers : Regression, SVM, Sigmoid, etc
* Fully Connected Neural Networks for Image Upscaling
* CNNs for image Classification
* Object detection using Yolo v1 loss and yolo v3 like architecture
* Diffusion Models trained on MNIST Digit dataset for digit generation
  * Unconditional
  * Time Conditioned
  * Class and Time Conditioned

---

Course Project : 
* Denoising Diffusion Restoration Models : Paper Implementation
*   The Idea behind this paper is to extend the functionality of a fully trained Diffusion Model to perform Image restoration tasks without any additional training
*   The proposed logic is applied during the sampling process of diffusion, and is capable of undoing Linear Inverse Degradation in images
*   I had attempted to train a diffusion model from scratch and use the author's logic to test the images on low resolution images [ like model trained on CIFAR100 Dataset ]
