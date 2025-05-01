# [CS444 : Deep Learning for Computer Vision at UIUC](https://slazebni.cs.illinois.edu/spring25/)
### The MPs and Project were done on individual efforts and no help was taken
---

The five MPs are listed as follows : 
* [Simple Linear Classifiers : Regression, SVM, Sigmoid, etc](https://github.com/girish-48/CS-444/tree/main/Linear%20classifiers)
* [Fully Connected Neural Networks for Image Upscaling](https://github.com/girish-48/CS-444/tree/main/MLP%20for%20image%20upscaling)
* [CNNs for image Classification](https://github.com/girish-48/CS-444/tree/main/CNN%20for%20Image%20classification)
* [Object detection using Yolo v1 loss and yolo v3 like architecture](https://github.com/girish-48/CS-444/tree/main/Object%20Detection%20using%20Yolo)
* Diffusion Models trained on MNIST Digit dataset for digit generation
  * Unconditional
  * Time Conditioned
  * Class and Time Conditioned

---

Course Project : 
* [Denoising Diffusion Restoration Models : Paper Implementation](https://proceedings.neurips.cc/paper_files/paper/2022/file/95504595b6169131b6ed6cd72eb05616-Paper-Conference.pdf)
  * The Idea behind this paper is to extend the functionality of a fully trained Diffusion Model to perform Image restoration tasks without any additional training
  * The proposed logic is applied during the sampling process of diffusion, and is capable of undoing Linear Inverse Degradation in images
  * I had attempted to train a diffusion model from scratch and use the author's logic to test the images on low resolution images [ like model trained on CIFAR100 Dataset ]
