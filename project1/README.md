# Week 5 Mini-Project
Please read through all the documentation before you get started.
This is a **MANDATORY** partner project. You and your assigned partner will submit one assignment.

## Self Supervised Learning for Image Classification

In this Mini-Project you will be implementing a paper that uses geometric transformations to extract features of an image without requiring these images to be labeled. This project will be for the most part from scratch; however, feel free to use the documentation below or reach out if you are confused. Although there is some skeleton code, feel free to delete all of it and implement this paper in whichever way makes the most sense to you. 

You have **two weeks** to complete the project and there will be regular checkpoints as outlined below. Good luck!

### Project Objectives
In this project you will learn the following valuable skills as discussed in lecture:
1. Using tensorflow data API to load and preprocess data
2. Using tensorboard to visualize training curves
3. Training a model from scratch and frequently checkpointing models
4. Implement good software engineering skills including the use of virtual environments, git (for partner work), and OOP

### Project Checkpoints
We will be regularly checking in on the following checkpoints so stay on top of your work!
1. Read this paper by 10/26. https://arxiv.org/pdf/1803.07728.pdf
2. Load and generate the rotation dataset. Start learning how to implement Tensorflow data loaders. Also make sure that you have your AWS account setup. Date: 10/29
3. Complete the model architecture (resnet18) and training loop. Date: 11/2
4. Debug model and complete training and show results. Date: 11/6
5. Completed project with results is due on Wednesday, November 6th at 8pm SHARP. NO EXCEPTIONS ON THIS DEADLINE.

### Setting up your environment
`pip3 install vitrualenv` (if not already installed)
`virtualenv venv`
`source venv/bin/activate`
`pip3 install -r requirements.txt`

To deactivate the environment you are in run:
`source deactivate`

### Code Structure
You will be writing code in the data.py, resnet.py, and rotnet.py.

`Rotnet.py` will contain the training loop and the basic graph for the model. You can start here to get a general idea of the flow of the code base.
`Resnet.py` will contain your implementation of the resnet model (you can find the architecture online or in the paper) and will be called from `rotnet.py`
`Data.py` will contain all your data loading functions.

Once you have implemented the model, you can start training by running `main.py` with the following command:
`python3 main.py --config config.yaml --train --data_dir ./data/cifar-10-batches-py/ --model_number 1`

`config.yaml` contains the configuration file with all the hyperparameters. If you have time, feel free to change these values and see how your model performs.

### Additional Details
#### Downloading the CIFAR-10 dataset
You can read more about the CIFAR-10 dataset here: https://www.kaggle.com/c/cifar-10
1. Go to this link https://www.cs.toronto.edu/~kriz/cifar.html
2. Right click on "CIFAR-10 python version" and click "Copy Link Address"
3. Go to your CLI and go into the `data` directory.
4. Run this cURL command to start downloading the dataset: `curl -O <URL of the link that you copied>`
5. To extract the data from the .tar file run: `tar -xzvf <name of file>` (type `man tar` in your CLI to see the different options for running the tar command).
**NOTE**: Each file in the directory contains a batch of images in CIFAR-10 that have been serialized using python's pickle module. You will have to first unpickle the data before loading it into your model.

#### Working with the dataset
Please consider the format that the data is in and what format you will need to convert it into in order to train your model. If you are stuck on this, feel free to use Google.
Your implementation for getting and loading the data from the model will likely differ from the implementation that I used so feel free to deviate from the skeleton code here.

#### Using Tensorflow dataloaders
We use tensorflow to build efficient datapipelines. You can read more about them here.
Resource: https://www.tensorflow.org/guide/datasets
This is an excellent Medium article on the different types of iterators and how to use them: https://medium.com/ymedialabs-innovation/how-to-use-dataset-and-iterators-in-tensorflow-with-code-samples-3bb98b6b74ab

#### Resnet18 Architecture
https://www.google.com/search?q=resnet+architecture&tbm=isch&source=iu&ictx=1&fir=nrwHYuY3M7ZNXM%253A%252CmlG8I6OjyTBN4M%252C_&vet=1&usg=AI4_-kRZVFcZ9REeELvn4BDXDpOJhFpNQg&sa=X&ved=2ahUKEwjd5NiphYjkAhVPKa0KHROtD3QQ9QEwBHoECAYQCQ#imgrc=eLRQQc-BgrBkxM:&vet=1

#### Saving and Restoring Models
Here is an excellent guide on how to save and restore models in Tensorflow
https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/

