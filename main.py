# Learning the Enigma with Recurrent Neural Networks
# Sam Greydanus. January 2017. MIT License.

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import os

from stacked_rnn import StackedRNN

MEM_SIZES = [3000,1024,512] # use for enigma
#MEM_SIZES = [32, 64, 128, 256, 512] # use for vigenere and autokey ciphers

tf.app.flags.DEFINE_bool("train", True, "Run the train loop (else eval model)")
tf.app.flags.DEFINE_bool("vary_mem", False, "Train this model repeatedly for different memory sizes")
tf.app.flags.DEFINE_integer("key_len", 6, "Maximum length of key for encoding/decoding message")
tf.app.flags.DEFINE_integer("tsteps", 20, "Number of timesteps for backpropagation")
tf.app.flags.DEFINE_integer("rnn_size", 256, "Number of hidden units in the rnn")
tf.app.flags.DEFINE_integer("ncells", 1, "Number of recurrent cells to stack")
tf.app.flags.DEFINE_integer("batch_size", 50, "Size of batch in minibatch gradient descent")
tf.app.flags.DEFINE_integer("save_every", 5000, "Save model after this number of train steps")
tf.app.flags.DEFINE_integer("total_steps", 250000, "Total number of training steps")
tf.app.flags.DEFINE_integer("print_every", 100, "Print training info after this number of train steps")
tf.app.flags.DEFINE_integer("acc_every", 500, "Print/save accuracy info after this number of train steps")
tf.app.flags.DEFINE_float("dropout", 1.0, "Dropout for the last (full-connected) layer")
tf.app.flags.DEFINE_float("lr", 5e-4, "Learning rate (alpha) for the model")
tf.app.flags.DEFINE_string("cipher", "vigenere", 'Type of cipher to solve. One of "vigenere", "autokey", or "enigma"')
tf.app.flags.DEFINE_string("A", "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "Alphabet to use for polyalphabetic cipher")
FLAGS = tf.app.flags.FLAGS

##### interpret user input #####
ciphers = ["vigenere", "autokey", "enigma", "crack-vigenere", "crack-autokey"]
if FLAGS.cipher not in ciphers:
	raise NotImplementedError('only {} ciphers are implemented'.format(ciphers))

CRACK_MODE = FLAGS.cipher not in ciphers[:3]
print("Crack mode: {}".format(CRACK_MODE))
if CRACK_MODE:
	from crack_train_utils import *
else:
	from train_utils import *

##### make cipher dataloader #####
data = None
if FLAGS.cipher == ciphers[0]:
	from vigenere import Vigenere
	data = Vigenere(FLAGS.A, tsteps=FLAGS.tsteps, key_len=FLAGS.key_len)
elif FLAGS.cipher == ciphers[1]:
	from autokey import Autokey
	data = Autokey(FLAGS.A, tsteps=FLAGS.tsteps, key_len=FLAGS.key_len)
elif FLAGS.cipher == ciphers[2]:
	print("Note: you must run this in Python 2 because Python 3 does not have the crypto_enigma module yet.")
	from enigma import Enigma
	data = Enigma(FLAGS.A, tsteps=FLAGS.tsteps, key_len=FLAGS.key_len) # only supports keylengths of 3
elif FLAGS.cipher == ciphers[3]:
	from crack_vigenere import CrackVigenere
	data = CrackVigenere(FLAGS.A, tsteps=FLAGS.tsteps, key_len=FLAGS.key_len)
elif FLAGS.cipher == ciphers[4]:
	from crack_autokey import CrackAutokey
	data = CrackAutokey(FLAGS.A, tsteps=FLAGS.tsteps, key_len=FLAGS.key_len)

def get_model(FLAGS):
	global CRACK_MODE
	model = StackedRNN(FLAGS=FLAGS, crack_mode=CRACK_MODE)
	model.count_params()
	return model

def eval_model(FLAGS, model, data, log):
	# evaluate model (quantitatively first, then qualitatively)
	print('plotting (check ./{} directory)...'.format(FLAGS.meta_dir))
	plot_stats(log, FLAGS)

	global_step = model.try_load_model()
	print( "accuracy is: {:3f}%".format(accuracy(model,data)) )
	sample(model, data, FLAGS)

# train model
if not FLAGS.vary_mem and FLAGS.train:

	# make bookkeping devices
	FLAGS.meta_dir = 'meta/' + FLAGS.cipher + '/' # directory to save loss history, figures, etc.
	FLAGS.save_dir = 'save/' + FLAGS.cipher + '/' # directory to save model
	os.makedirs(FLAGS.save_dir) if not os.path.exists(FLAGS.save_dir) else None
	os.makedirs(FLAGS.meta_dir) if not os.path.exists(FLAGS.meta_dir) else None
	log = Logger(FLAGS)

	# train model
	model = get_model(FLAGS)
	train(model, data, log, FLAGS)
	
elif FLAGS.train:
	for m in MEM_SIZES:
		FLAGS.rnn_size = m # change rnn memory size

		# make bookkeping devices
		FLAGS.meta_dir = 'meta/' + FLAGS.cipher + '-{}/'.format(m) # directory to save loss history, figures, etc.
		FLAGS.save_dir = 'save/' + FLAGS.cipher + '-{}/'.format(m) # directory to save model
		os.makedirs(FLAGS.save_dir) if not os.path.exists(FLAGS.save_dir) else None
		os.makedirs(FLAGS.meta_dir) if not os.path.exists(FLAGS.meta_dir) else None
		log = Logger(FLAGS)

		# train as usual
		model = get_model(FLAGS)
		train(model, data, log, FLAGS)
		eval_model(FLAGS, model, data, log)

elif not FLAGS.vary_mem and not FLAGS.train:
	# make bookkeping devices
	FLAGS.meta_dir = 'meta/' + FLAGS.cipher + '/' # directory to save loss history, figures, etc.
	FLAGS.save_dir = 'save/' + FLAGS.cipher + '/' # directory to save model
	os.makedirs(FLAGS.save_dir) if not os.path.exists(FLAGS.save_dir) else None
	os.makedirs(FLAGS.meta_dir) if not os.path.exists(FLAGS.meta_dir) else None

	print(FLAGS.meta_dir) ; print(FLAGS.save_dir)
	log = Logger(FLAGS)
	model = get_model(FLAGS)
	eval_model(FLAGS, model, data, log)
