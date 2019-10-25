import tensorflow as tf
import sys

class ResNet():
    def __init__(self, training=True):
        self.training = training

    def add_residual_block(self, input, block_number, in_channels, out_channels):
        block_number = str(block_number) #This was used for providing a unqiue name to each layer.
        skip = tf.identity(input)

        if in_channels != out_channels:
            #TODO: perform 1x1 convolution to match output dimensions for skip connection
            ...

        #TODO: Implement one residual block (Convolution, batch_norm, relu)
        ...

        #TODO: Add the skip connection and ReLU the output
        ...
        return

    def forward(self, data):
        #TODO: 64 7x7 convolutions followed by batchnorm, relu, 3x3 maxpool with stride 2
        ...

        #TODO: Add residual blocks of the appropriate size. See the diagram linked in the README for more details on the architecture.
        # Use the add_residual_block helper function
        ...

        #TODO: perform global average pooling on each feature map to get 4 output channels
        ...

        return logits

    def add_convolution(self,
                        input,
                        name,
                        filter_size,
                        input_channels,
                        output_channels,
                        padding):
        #TODO: Implement a convolutional layer with the above specifications
        ...
        return
