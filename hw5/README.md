# Week 5 Mini-Project
## Self Supervised Learning for Image Classification

In this Mini-Project you will be implementing a paper that uses geometric transformations to extract features of an image without requiring these images to be labeled. This project will be for the most part from scratch; however, feel free to use the documentation below or reach out to club members to guide you. You have two weeks to complete the project and there will be regular checkpoints as outlined below. Good luck!

### Project Objectives
In this project you will learn the following valuable skills:
1. Using tensorflow data API to load and preprocess data
2. Utilizing CPUs and GPUs for async processing
3. Using tensorboard to visualize training curves
4. Training a model from scratch and frequently checkpointing models
5. Visualizing the features of your network
6. Implement good software engineering skills including the use of virtual environments and OOP

### Project Checkpoints
1. Read this paper by the Tuesday after lecture. https://arxiv.org/pdf/1803.07728.pdf
2. Load and generate the rotation dataset. Start learning how to implement Tensorflow data loaders. Also make sure that you have your AWS account setup. Date: Thursday
3. Complete the model architecture and training loop. Date: Saturday
4. Debug mode and complete training and show results. Date: Wednesday (Please show up to Sunday OH to get help debugging your model)
5. Completed project with feature visualization and results is due on Monday at 8pm SHARP. NO EXCEPTIONS ON THIS DEADLINE.

### Setting up your environment
`pip3 install vitrualenv` (if not already installed)
`virtualenv venv`
`source venv/bin/activate`
`pip3 install -r requirements.txt`


To deactivate the environment you are in run:
`source deactivate`

### Downloading the CIFAR-10 dataset
#### You can read more about the CIFAR-10 dataset here: https://www.kaggle.com/c/cifar-10
1. Go to this link https://www.cs.toronto.edu/~kriz/cifar.html
2. Right click on "CIFAR-10 python version" and click "Copy Link Address"
3. Go to your CLI and go into the `data` directory.
4. Run this cURL command to start downloading the dataset: `curl -O <URL of the link that you copied>`
5. To extract the data from the .tar file run: `tar -xzvf <name of file>` (type `man tar` in your CLI to see the different options for running the tar command).
**NOTE**: Each file in the directory contains a batch of images in CIFAR-10 that have been serialized using python's pickle module. You will have to first unpickle the data before loading it into your model.

### Working with the dataset
Please consider the format that the data is in and what format you will need to convert it into in order to train your model. If you are stuck on this, feel free to use Google.

### Using Tensorflow dataloaders
Resource: https://www.tensorflow.org/guide/datasets
This is an excellent Medium article on the different types of iterators and how to use them: https://medium.com/ymedialabs-innovation/how-to-use-dataset-and-iterators-in-tensorflow-with-code-samples-3bb98b6b74ab

### Resnet18 Architecture
https://www.google.com/search?q=resnet+architecture&tbm=isch&source=iu&ictx=1&fir=nrwHYuY3M7ZNXM%253A%252CmlG8I6OjyTBN4M%252C_&vet=1&usg=AI4_-kRZVFcZ9REeELvn4BDXDpOJhFpNQg&sa=X&ved=2ahUKEwjd5NiphYjkAhVPKa0KHROtD3QQ9QEwBHoECAYQCQ#imgrc=eLRQQc-BgrBkxM:&vet=1

### Saving and Restoring Models
Here is an excellent guide on how to save and restore models in Tensorflow
https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
