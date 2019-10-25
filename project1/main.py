import yaml
import argparse
import tensorflow as tf
from rotnet import RotNet

parser = argparse.ArgumentParser(description='Configuration details for training/testing rotation net')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--train', action='store_true')
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--image', type=str)
parser.add_argument('--model_number', type=str, required=True)

args = parser.parse_args()

def main():
    with tf.compat.v1.Session() as sess:
        rotation_net = RotNet(sess, args)
        if args.train:
            rotation_net.train()
        else:
            rotation_net.predict(args.image)

if __name__ == "__main__":
    main()
