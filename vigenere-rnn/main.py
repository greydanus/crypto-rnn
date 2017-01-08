# Decoding the Vigenere Cipher with Recurrent Neural Networks
# Sam Greydanus. January 2017. MIT License.

import numpy as np
import tensorflow as tf

from stacked_rnn import StackedRNN
from dataloader import Dataloader
from train_utils import *

tf.app.flags.DEFINE_bool("train", True, "Run the train loop (else eval model)")
tf.app.flags.DEFINE_integer("key_len", 6, "Maximum length of key for encoding/decoding message")
tf.app.flags.DEFINE_integer("tsteps", 15, "Number of timesteps for backpropagation")
tf.app.flags.DEFINE_integer("rnn_size", 200, "Number of hidden units in the rnn")
tf.app.flags.DEFINE_integer("ncells", 1, "Number of recurrent cells to stack")
tf.app.flags.DEFINE_integer("batch_size", 32, "Size of batch in minibatch gradient descent")
tf.app.flags.DEFINE_integer("turnover", 5, "Number of times to reuse a synthesized training example")
tf.app.flags.DEFINE_integer("replay_queue_depth", 500, "Size of replay-like dataset to hold synthesized data")
tf.app.flags.DEFINE_integer("save_every", 5000, "Save model after this number of train steps")
tf.app.flags.DEFINE_integer("print_every", 100, "Print training info after this number of train steps")
tf.app.flags.DEFINE_float("dropout", 1.0, "Dropout for the last (full-connected) layer")
tf.app.flags.DEFINE_float("lr", 5e-4, "Learning rate (alpha) for the model")
tf.app.flags.DEFINE_string("save_path", "models/model.ckpt", "Directory in which to save model")
tf.app.flags.DEFINE_string("A", "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "Alphabet to use for polyalphabetic cipher")
FLAGS = tf.app.flags.FLAGS

dataloader = Dataloader(FLAGS.A, tsteps=FLAGS.tsteps, max_key_len=FLAGS.key_len) # class for synthesizing data
model = StackedRNN(xlen=len(FLAGS.A) + 1, ylen=len(FLAGS.A) + 1, FLAGS=FLAGS) # model for analyzing the data
print "="*5 + " COUNTING MODEL PARAMETERS " + "="*5
model.count_params()
print "="*37

# train model
if FLAGS.train: train(model, dataloader, FLAGS)

# evaluate model (quantitatively first, then qualitatively)
global_step = model.try_load_model()
print "accuracy is: {:3f}%".format(accuracy(model,dataloader))
sample(model, dataloader)