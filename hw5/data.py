import tensorflow as tf
import numpy as np
import pickle
import cv2
from PIL import Image
import imutils

class Data(object):
    def __init__(self, data_dir, height, width, batch_size):
        self.data_dir = data_dir
        self.height = height
        self.width = width
        self.batch_size = batch_size

    def get_rot_data_iterator(self, images, labels):
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        dataset = dataset.shuffle(buffer_size=10000, reshuffle_each_iteration=True)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.repeat()
        dataset = dataset.map(self.preprocess, num_parallel_calls=2)
        dataset = dataset.prefetch(self.batch_size)
        iterator = dataset.make_initializable_iterator()
        return iterator

    def get_training_data(self):
        images = []
        labels = []
        for i in range(1, 6):
            data = self._get_next_batch_from_file(i)
            images.extend(list(self.convert_images(data[b"data"])))
            labels.extend(list(data[b"labels"]))
        return np.array(images), tf.keras.utils.to_categorical(np.array(labels))

    def convert_images(self, raw_images):
        images = raw_images / 255.0
        images = raw_images.reshape([-1, 3, self.height, self.width])
        images = images.transpose([0, 2, 3, 1])
        return images

    def _get_next_batch_from_file(self, batch_number):
        data = self._unpickle_data(self.data_dir + self._get_batch_name(batch_number))
        return data

    def _get_batch_name(self, number):
        return "data_batch_{0}".format(number)

    def _unpickle_data(self, filepath):
        with open(filepath, 'rb') as data:
            dict = pickle.load(data, encoding='bytes')
        return dict

    def get_test_data(self):
         data = self._unpickle_data(self.data_dir + "test_batch")
         return data[b"data"], tf.keras.utils.to_categorical(data[b"labels"])

    def preprocess(self, images, labels):
        rot_labels = []
        rot_images = []
        rotations = [90, 180, 270]
        for image in images:
            rot_labels.append(0)
            rot_images.append(image)
            for i, angle in enumerate(rotations, 1):
                rotated = imutils.rotate_bound(image, angle)
                rot_images.append(rotated)
                rot_labels.append(i)

        return np.array(rot_images), tf.keras.utils.to_categorical(np.array(rot_labels))

    @staticmethod
    def print_image_to_screen(data):
        """
        Used for debugging purposes.
        """
        img = Image.fromarray(data, 'RGB')
        img.show()

    @staticmethod
    def get_image(image_path):
        return

if __name__ == "__main__":
    DATA_DIR = "./data/cifar-10-batches-py/"
    data_obj = Data(DATA_DIR, 32, 32, 5000)
    x, y = data_obj.get_training_data()
    xr, yr = data_obj.preprocess(x, y)
