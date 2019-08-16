import tensorflow as tf

class ResNet():
    def __init__(self, training=True):
        self.training = training

    def add_residual_block(self, input, block_number, in_channels, out_channels, downsample=False):
        if downsample:
            skip = tf.nn.max_pool(input, [1, 2, 2, 1], [1, 2, 2, 1], "VALID")
        else:
            skip = tf.identity(input)

        if in_channels != out_channels:
            #perform 1x1 convolution to match output dimensions
            skip = self.add_convolution(skip, "Wconv2_" + block_number, 1, in_channels, out_channels, "SAME")
        layer = self.add_convolution(input, "Wconv0_" + block_number, 3, in_channels, out_channels, "SAME")
        layer = tf.layers.batch_normalization(layer, training=self.training)
        layer = tf.nn.relu(layer)
        layer = self.add_convolution(layer, "Wconv1_" + block_number, 3, in_channels, out_channels, "SAME")
        layer = tf.layers.batch_normalization(layer, training=self.training)
        output = layer + skip
        return tf.nn.relu(output)

    def forward(self, data):
        #7x7 convolution followed by batchnorm, relu, maxpool
        layer = self.add_convolution(data, "Wconv0_0", 7, 3, 64, "SAME")
        layer = tf.layers.batch_normalization(layer, training=self.training)
        layer = tf.nn.relu(layer)
        layer = tf.nn.max_pool(layer, [1, 3, 3, 1], [1, 2, 2, 1], "VALID")

        #8 layers of residual blocks
        layer = self.residual_block(layer, 1, 64, 64)
        layer = self.residual_block(layer, 2, 64, 64)

        layer = self.residual_block(layer, 3, 64, 128, downsample=True)
        layer = self.residual_block(layer, 4, 128, 128)

        layer = self.residual_block(layer, 5, 128, 256, downsample=True)
        layer = self.residual_block(layer, 6, 256, 256)

        layer = self.residual_block(layer, 7, 256, 512, downsample=True)
        layer = self.residual_block(layer, 8, 512, 512)

        layer = self.add_convolution(layer, "Wconvf", 1, 512, 4, "SAME")
        #perform global average pooling on each feature map
        logits = tf.reduce_mean(layer, axis=[1,2])
        return logits

    def add_convolution(self,
                        input,
                        name,
                        filter_size,
                        input_channels,
                        output_channels,
                        padding):
        shape = [filter_size, filter_size, input_channels, output_channels]
        Wconv = tf.get_variable(name, shape=shape)
        layer = tf.nn.conv2d(input, Wconv, padding=padding)
        return layer
