import tensorflow as tf

class ResNet():
    def residual_block(self):
        return

    def forward(self, data):
        Wconv = tf.get_variable("Wconv", shape=[3, 3, 3, 64])
        layer = tf.nn.conv2d(data, Wconv, padding="SAME")
        layer = 



        return logits
