import yaml
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.client import device_lib
from resnet import ResNet
from data import Data

class RotNet(object):
    def __init__(self, sess, args):
        print("[INFO] Reading configuration file.")
        self.config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

        self.sess = sess
        self.data_dir = args.data_dir
        self.model_dir = "./checkpoints/"
        self.classes = ["0", "90", "180", "270"]
        self.model_number = args.model_number

        self._populate_model_hyperparameters()
        self.training_data_obj = Data(self.data_dir,
                            batch_size=self.batch_size,
                            height=self.height,
                            width=self.width
                            )
        self.build_base_graph()

        if args.train:
            self.build_train_graph()

        print(device_lib.list_local_devices())

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
        self.y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 10])

        self.iterator = self.training_data_obj.get_rot_data_iterator(self.X, self.y)
        data_X, data_y = self.iterator.get_next()

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

    def train(self):
        self.sess.run(tf.compat.v1.global_variables_initializer())

        X, y = self.training_data_obj.get_training_data()
        print(X.shape)
        print(y.shape)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15 , shuffle=True)

        #training_iterator = data_obj.get_rot_data_iterator(X_train, y_train)
        #validation_iterator = data_obj.get_rot_data_iterator(X_val, y_val)

        num_batches = int(len(X_train) / self.batch_size)
        global_step = 0
        print("[INFO] Starting Training...")
        for epoch in range(self.num_epochs):
            self.sess.run(self.iterator.initializer, feed_dict = {self.X: X_train, self.y: y_train})
            for batch in range(num_batches):
                self._update_learning_rate(epoch)
                _, accuracy = self.sess.run([self.opt, self.acc])
                global_step += 1
                print("Epoch: {0}, Batch: {1} ==> Accuracy: {2}".format(epoch, batch, accuracy))

            self.sess.run(self.iterator.initializer, feed_dict = {self.X: X_val, self.y: y_val})
            loss, accuracy = self.sess.run([self.loss, self.acc])
            print("(Validation) Epoch: {0} ===> Accuracy: {2}".format(epoch, accuracy))
            self.save_checkpoint(global_step, epoch)

        # test_dataset = Data(self.data_dir,
        #             batch_size=self.batch_size,
        #             height=self.height,
        #             width=self.width
        #             )
        # loss, accuracy = sess.run([self.loss, self.acc], feed_dict={})
        # print("Test Accuracy: ", accuracy)

    # def predict(self, image_path):
    #     data = Data.get_image(image_path)
    #     pred = self.sess.run([self.predictions], feed_dict{X: data, y=None})
    #     return self.classes[pred]

    def restore_from_checkpoint(self):
        """
        TODO
        """
        return

    def save_checkpoint(self, global_step, epoch):
        self.saver.save(self.sess,
                        self.model_dir + "model{0}/model.ckpt".format(self.model_number),
                        global_step=global_step,
                        write_meta_graph=(epoch==0))

    def _update_learning_rate(self, epoch):
        if epoch == 80 or epoch == 60 or epoch == 30:
            self.learning_rate = self.learning_rate * 0.2
