import yaml
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.client import device_lib
from resnet import ResNet
from data import Data

class RotNet(object):
    def __init__(self, sess, args):
        print("[INFO] Reading configuration file")
        self.config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

        self.sess = sess
        self.data_dir = args.data_dir
        self.model_dir = "./checkpoints/"
        self.classes = ["0", "90", "180", "270"]
        self.model_number = args.model_number

        self._populate_model_hyperparameters()
        self.data_obj = Data(self.data_dir,
                            batch_size=self.batch_size,
                            height=self.height,
                            width=self.width
                            )
        self.build_base_graph()

        if args.train:
            self.build_train_graph()

        print(device_lib.list_local_devices())
        self.summary = tf.summary.merge_all()

    def _populate_model_hyperparameters(self):
        self.batch_size = self.config["batch_size"]
        self.weight_decay = self.config["weight_decay"]
        self.momentum = self.config["momentum"]
        self.learning_rate = self.config["learning_rate"]
        self.height = self.config["image_height"]
        self.width = self.config["image_width"]
        self.num_epochs = self.config["num_epochs"]

    def build_base_graph(self):
        self.X = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3])
        self.y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, len(self.classes)])
        self.bs = tf.compat.v1.placeholder(dtype=tf.int64, shape=[])

        self.iterator = self.data_obj.get_rot_data_iterator(self.X, self.y, self.bs)
        data_X, data_y = self.iterator.get_next()
        img_sum = tf.summary.image('train_images', data_X)
        with tf.device('/cpu:0'):
            model = ResNet()
            self.logits = model.forward(data_X)
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=data_y))
            self.prob = tf.nn.softmax(self.logits)
            self.predictions = tf.cast(tf.argmax(self.prob, axis=1), tf.float32)
            actual = tf.cast(tf.argmax(data_y, axis=1), tf.float32)
            self.acc = tf.reduce_mean(tf.cast(tf.equal(self.predictions, actual), tf.float32))

    def build_train_graph(self):
        with tf.device('/cpu:0'):
            self.opt = tf.compat.v1.train.RMSPropOptimizer(learning_rate=self.learning_rate,
                                            decay=self.weight_decay,
                                            momentum=self.momentum).minimize(self.loss)
            self.saver = tf.compat.v1.train.Saver(max_to_keep=5)

        if os.path.exists("./checkpoints/model{0}".format(self.model_number)):
            self.start_epoch = self.restore_from_checkpoint()
        else:
            self.start_epoch = 0

        self.train_writer = tf.summary.FileWriter("./logs/train", self.sess.graph)

    def train(self):
        self.sess.run(tf.compat.v1.global_variables_initializer())

        X, y = self.data_obj.get_training_data()
        print(X.shape)
        print(y.shape)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15 , shuffle=True)

        num_batches = int(len(X_train) / self.batch_size)
        global_step = 0
        print("[INFO] Starting Training...")
        for epoch in range(self.start_epoch, self.num_epochs):
            global_step += 1
            self.sess.run(self.iterator.initializer, feed_dict = {self.X: X_train, self.y: y_train, self.bs: self.batch_size})
            for batch in range(num_batches):
                self._update_learning_rate(epoch)
                _, loss, accuracy, summary = self.sess.run([self.opt, self.loss, self.acc, self.summary])
                self.train_writer.add_summary(summary)
                print("Epoch: {0}, Batch: {1} ==> Accuracy: {2}, Loss: {3}".format(epoch, batch, accuracy, loss))

            self.sess.run(self.iterator.initializer, feed_dict = {self.X: X_val, self.y: y_val, self.bs: len(X_val)})
            loss, accuracy = self.sess.run([self.loss, self.acc])
            print("(Validation) Epoch: {0} ===> Accuracy: {1}".format(epoch, accuracy))
            self.save_checkpoint(global_step, epoch)

        X_test, y_test = self.data_obj.get_test_data()
        self.sess.run(self.iterator.initializer, feed_dict = {self.X: X_train, self.y: y_train, self.bs: len(X_train)})
        loss, accuracy = self.sess.run([self.loss, self.acc])
        print("Test Accuracy: ", accuracy)

    def predict(self, image_path):
        """
        TODO
        Gets the latest models and
        """
        self.restore_from_checkpoint()
        self.sess.run(tf.compat.v1.global_variables_initializer())
        image = self.data_obj.load_image(image_path)
        self.sess.run(self.iterator.initializer, feed_dict = {self.X: image, self.y: [0], self.bs: -1})
        prediction = self.sess.run([self.predictions], feed_dict={})
        return self.classes[prediction[0]]

    def restore_from_checkpoint(self):
        print("[INFO] Restoring model {0} from latest checkpoint".format(self.model_number))
        checkpoint_dir = "./checkpoints/model{0}".format(self.model_number)
        checkpoint = tf.compat.v1.train.latest_checkpoint(checkpoint_dir)
        print("[DEBUG] Latest checkpoint file:", checkpoint)
        start_epoch = 1 + int(checkpoint.split('.ckpt')[1].split('-')[1])
        self.saver.restore(self.sess, checkpoint)
        return start_epoch

    def save_checkpoint(self, global_step, epoch):
        self.saver.save(self.sess,
                        self.model_dir + "model{0}/model.ckpt".format(self.model_number),
                        global_step=global_step,
                        write_meta_graph=(epoch==0))

    def _update_learning_rate(self, epoch):
        if epoch == 80 or epoch == 60 or epoch == 30:
            self.learning_rate = self.learning_rate * 0.2
