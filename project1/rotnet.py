import yaml
import os
import sys
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.client import device_lib
from resnet import ResNet
from data import Data

class RotNet(object):
    def __init__(self, sess, args):
        #TODO: Look through this function to see which attributes have already been initalized for you.
        print("[INFO] Reading configuration file")
        self.config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

        self.sess = sess
        self.data_dir = args.data_dir
        self.model_dir = "./checkpoints/"
        self.classes = ["0", "90", "180", "270"]
        self.model_number = args.model_number
        self.model = ResNet()

        self._populate_model_hyperparameters()
        self.data_obj = Data(self.data_dir,
                            batch_size=self.batch_size,
                            height=self.height,
                            width=self.width
                            )
        self.build_base_graph()

        if args.train:
            #If we are training, then we want to run the optimizer
            self.build_train_graph()

        #List the compute available on the device that this script is being run on.
        print(device_lib.list_local_devices())

        #This collects the add_summary operations that you defined in the graph. You should be saving your metrics to self.summary
        self.summary = tf.summary.merge_all()

    def _populate_model_hyperparameters(self):
        """
        This is a helper function for populating the hyperparameters from the yaml file
        """
        self.batch_size = self.config["batch_size"]
        self.weight_decay = self.config["weight_decay"]
        self.momentum = self.config["momentum"]
        self.learning_rate = self.config["learning_rate"]
        self.height = self.config["image_height"]
        self.width = self.config["image_width"]
        self.num_epochs = self.config["num_epochs"]

    def build_base_graph(self):
        #TODO: Initialize your dataloader here using tf.data by calling "get_rot_data_iterator"
        ...
        data_X, data_y = self.iterator.get_next()

        #TODO: Construct the Resnet in resnet.py
        logits = self.model.forward(data_X)

        #TODO: Calculate the loss and accuracy from your output logits.
        # Add your accuracy metrics and loss to the tensorboard summary using tf.summary
        ...

        #END OF FUNCTION

    def build_train_graph(self):
        #TODO: Create an optimizer that minimizes the loss function that you defined in the above function
        ...

        #This will restore a model @ the latest epoch if you have already started training
        #If it cannot find a checkpoint, it will set the starting epoch to zero
        if os.path.exists("./checkpoints/model{0}".format(self.model_number)):
            #TODO: Complete the restore from checkpoint function
            self.start_epoch = self.restore_from_checkpoint()
        else:
            self.start_epoch = 0

        #Creates a writer for Tensorboard
        self.train_writer = tf.summary.FileWriter("./logs/train/" + str(self.model_number), self.sess.graph)

    def train(self):
        #TODO: Initialize your graph variables
        ...

        #TODO: Implement and call the get_training_data function to get the data from disk
        #NOTE: Depending on how you implement your iterator, you may not need to load the data here.
        ...

        #TODO: Split the data into a training and validation set: see sklearn train_test_split
        ...


        #TODO: Implement the training and validation loop and checkpoint your file at each epoch
        print("[INFO] Starting Training...")
        for epoch in range(self.start_epoch, self.num_epochs):
            for batch in range(num_batches):
                self._update_learning_rate(epoch)
                ...

                #TODO: Make sure you are using the tensorflow add_summary method to add the data for each batch to Tensorboard
                ...

                print("Epoch: {0}, Batch: {1} ==> Accuracy: {2}, Loss: {3}".format(epoch, batch, accuracy, loss))

            #TODO: Calculate validation accuracy and loss
            ...

            #TODO: Use the save_checkpoint method below to save your model weights to disk.
            ...

        #TODO: Evaluate your data on the test set after training
        ...

    def predict(self, image_path):
        #TODO: Once you have trained your model, you should be able to run inference on a single image by reloading the weights
        ...
        return

    def restore_from_checkpoint(self):
        #TODO: restore the weights of the model from a given checkpoint
        #this function should return the latest epoch from training (you can get this from the name of the checkpoint file)
        ...
        return

    def save_checkpoint(self, global_step, epoch):
        #TODO: This function should save the model weights. If we are on the first epoch it should also save the graph.
        ...
        return

    def _update_learning_rate(self, epoch):
        #In the paper the learning rate is updated after certain epochs to slow down learning.
        if epoch == 80 or epoch == 60 or epoch == 30:
            self.learning_rate = self.learning_rate * 0.2
