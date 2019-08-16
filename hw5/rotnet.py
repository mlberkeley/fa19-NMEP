import yaml
from sklearn.model_selection import train_test_split
import tensorflow as tf
from resnet import ResNet

class RotNet(object):
    def __init__(self, sess, args):
        config = yaml.load(open(args.config, 'r'))
        self.config = AttrDict(config)

        self.sess = sess
        self.data_dir = args.data_dir
        self.model_dir = "./checkpoints/"
        self.classes = ["0", "90", "180", "270"]
        self.model_number = args.model_number

        self._populate_model_hyperparameters()
        self.build_base_graph()

        if args.train:
            self.build_train_graph()

    def _populate_model_hyperparameters(self):
        self.batch_size = self.config.batch_size
        self.weight_decay = self.config.weight_decay
        self.momentum = self.config.momentum
        self.learning_rate = self.config.learning_rate

    def build_base_graph(self):
        with tf.device('/cpu:0'):
            handle = tf.placeholder(tf.string, shape=[])
            iterator = tf.data.Iterator.from_string_handle(handle, output_types=(tf.float32, tf.int32))
            X, y = iterator.get_next()

        with tf.device('/gpu:0'):
            model = ResNet()
            self.logits = model.forward(X)
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=y))
            self.prob = tf.nn.softmax(logits)
            self.predictions = tf.cast(tf.argmax(self.prob, axis=1), tf.float32)
            actual = tf.cast(tf.argmax(y, axis=1), tf.float32)
            self.acc = tf.reduce_mean(tf.cast(tf.equal(self.predictions, actual), tf.float32))

    def build_train_graph(self):
        with tf.device('/gpu:0'):
            lr = tf.placeholder(dtype=tf.float32, name="Learning Rate")
            self.opt = tf.RMSPropOptimizer(learning_rate=self.learning_rate,
                                            decay=self.weight_decay,
                                            momentum=self.momentum).minimize(self.loss)
            self.saver = tf.train.Saver(max_to_keep=5)

    def train(self):
        data_obj = Data(self.data_dir,
                    batch_size=self.batch_size,
                    height=self.height,
                    width=self.width
                    )

        X, y = data_obj.get_training_data()
        X_train, y_train, X_val, y_val = train_test_split(X, y, test_size=0.15 , shuffle=True)

        training_handle = data_obj.get_rot_data_iterator(X_train, y_train)
        validation_handle = data_obj.get_rot_data_iterator(X_val, y_val)

        num_batches = int(len(X_train) / self.batch_size)
        global_step = 0
        for epoch in range(self.num_epochs):
            for batch in range(num_batches):
                self._update_learning_rate(epoch)
                _, accuracy = sess.run([self.opt, self.acc], feed_dict={handle: training_handle})
                global_step += 1
                print("Epoch: {0}, Batch: {1} ==> Accuracy: {2}".format(epoch, batch, accuracy))
            loss, accuracy = sess.run([self.loss, self.acc], feed_dict={validation_handle})
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

    # def restore_from_checkpoint(self):
    #     return

    def save_checkpoint(self, global_step, epoch):
        self.saver.save(self.sess,
                        self.model_dir + "model{0}/model.ckpt".format(self.model_number),
                        global_step=global_step,
                        write_meta_graph=(epoch==0))

    def _update_learning_rate(self, epoch):
        if epoch == 80 or epoch == 60 or epoch == 30:
            self.learning_rate = self.learning_rate * 0.2
