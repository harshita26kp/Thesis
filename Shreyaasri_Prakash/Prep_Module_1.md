# RESOURCES

Starting point:

1. Pick an appropriate dataset
2. Gather papers
3. Set up GitHub
4. Set up a virtual Environment

Resources: (Understand all the 4 -> Deadline 10th July - Make sure theoretically you are strong(Q&A in the upcoming meeting))

1. https://youtu.be/QzY57FaENXg?si=BeI0yB9aVOplsSSO 
2. https://poloclub.github.io/cnn-explainer/
3. https://youtu.be/m0fmBY4t5BI?si=ZIUEfAxgT3xzvl4i
4. https://youtu.be/KuXjwB4LzSA?si=mv7Xq5-gmB5BYMPS

### Key Concepts:

- convolutional kernel
- feature map
- maxpool
- flatten



## Convolutional Neural Networks (Deadline 12th July)
Train a CNN on Fashion-Mnist data

the basic structure:
- preprocessing 
- input layer
- conv2D layer
- maxpool2D
- activation 
- flatten, followed by Dense

Task: Vary the following hyperparameters:

- size of kernels
- number of conv2D layers
- number of filters
- use model.summary() to see how the number of trainable parameters changes

For faster training, start with a smaller sample of training data points (1000) 

Python code for CNN beginners: 
fashion_mnist_cnn_.ipynb



## Pretrained Networks (Deadline 12th July)
Run predictions using MobileNet

### Key Concepts:

- what is imagenet competition 
- structure of VGG-16
- run a prediction on 2-3 images using MobileNet
- what shapes come out of the layers, the flattening especially
- run the model with `include_top=False` and see 
- compare models in https://keras.io/api/applications/

If you feel comfortable to go for a deeper dive, create variations of the images and see whether the model can deal with them, including:

- change brightness
- add blur
- insert black square
- remove one half
- shift the image
- rotate
- flip upside-down
- flip left-right
- blend two images together

FYI: The VGG-16 is an inefficient model compared to others in the competition, but it explains the core idea best.
The ImageNet models are historically important, there has been a lot of gradual improvement since then. Discuss what modern pretrained models (like Google Images) do better/differently.


https://www.academis.eu/machine_learning/deep_learning/convolutional_neural_networks/pretrained.html

Python code:

training_data (sent via teams)

PretrainedNetworks_1.ipynb



# Papers

2015_image_captioning.pdf
2015_yolo.pdf
