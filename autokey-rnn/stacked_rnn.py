# Decoding the Autokey Cipher with Recurrent Neural Networks
# Sam Greydanus. January 2017. MIT License.

import numpy as np
import tensorflow as tf

# stacked recurrent neural network with LSTM cells
class StackedRNN():
    def __init__(self, xlen, ylen, FLAGS):
        self.sess = tf.InteractiveSession()
        self.batch_size = batch_size = FLAGS.batch_size
        self.tsteps = tsteps = FLAGS.tsteps + FLAGS.key_len
        self.xlen = xlen
        self.ylen = ylen
        self.rnn_size = rnn_size = FLAGS.rnn_size
        self.fc1_size = fc1_size = FLAGS.rnn_size
        self.ncells = ncells = FLAGS.ncells
        self.layers = layers = [{}]*(ncells + 1)
        self.keep_prob = tf.placeholder(tf.float32)
        self.dropout = FLAGS.dropout
        self.lr = lr = FLAGS.lr
        self.x = x = tf.placeholder(tf.float32, shape=[None, None, xlen], name="x")
        self.y = y = tf.placeholder(tf.float32, shape=[None, None, ylen], name="y")
        self.scope = 'model'
        self.save_path = save_path = FLAGS.save_path

        rnn_init = tf.truncated_normal_initializer(stddev=0.075, dtype=tf.float32)
        xavier_dense = tf.truncated_normal_initializer(stddev=1./np.sqrt(rnn_size), dtype=tf.float32)

        with tf.variable_scope(self.scope,reuse=False):
            for i in range(ncells):
                cell = layers[i]
                cell['rnn'] = tf.contrib.rnn.LSTMCell(rnn_size, state_is_tuple=True, initializer=rnn_init)
                cell['istate_batch'] = cell['rnn'].zero_state(batch_size=batch_size, dtype=tf.float32)
                cell['istate'] = cell['rnn'].zero_state(batch_size=1, dtype=tf.float32)
            layers[-1]['W_fc1'] = tf.get_variable("W_fc1", [rnn_size, ylen], initializer=xavier_dense)

        self.y_hat_batch = y_hat_batch = self.forward(tsteps=tsteps, reuse=False)
        self.y_hat = self.forward(tsteps=1, reuse=True)
        
        self.loss = tf.nn.l2_loss(y - y_hat_batch) / (batch_size*self.tsteps)
        self.optimizer = tf.train.AdamOptimizer(lr)
        self.grads = self.optimizer.compute_gradients(self.loss, var_list=tf.trainable_variables())
        self.train_op = self.optimizer.apply_gradients(self.grads)

        self.sess.run(tf.global_variables_initializer())
        self.reset_states()
        
    def reset_states(self):
        for i in range(self.ncells):
            self.layers[i]['state_c'] = self.layers[i]['istate'].c.eval()
            self.layers[i]['state_h'] = self.layers[i]['istate'].h.eval()
            
    def forward(self, tsteps, reuse):
        with tf.variable_scope(self.scope, reuse=reuse):
            x = tf.reshape(self.x, [-1, self.xlen])
            hs = [tf.squeeze(h_, [1]) for h_ in tf.split(tf.reshape(x, [-1, tsteps, self.xlen]), tsteps, 1)]

            for i in range(self.ncells):
                state = self.layers[i]['istate'] if tsteps is 1 else self.layers[i]['istate_batch']
                cell = self.layers[i]['rnn']
                cell_scope = self.scope + '_cell' + str(i)
                hs, self.layers[i]['fstate'] = tf.contrib.legacy_seq2seq.rnn_decoder(hs, state, cell, scope=cell_scope)
            rnn_out = tf.reshape(tf.concat(hs, 1), [-1, self.rnn_size])
            rnn_out = tf.nn.dropout(rnn_out, self.keep_prob)

            logps = tf.matmul(rnn_out, self.layers[-1]['W_fc1'])
            logps = tf.nn.softmax(logps)
            p = tf.reshape(logps, [-1, tsteps, self.ylen])
        return p
    
    def train_step(self, batch):
        feed = {self.x: batch[0], self.y: batch[1], self.keep_prob: self.dropout}
        train_loss, _ = self.sess.run([self.loss, self.train_op], feed_dict=feed)
        return train_loss

    def decode(self, input):
        feed = {self.x: input, self.keep_prob: 1}
        y_hat = self.sess.run(self.y_hat_batch, feed_dict=feed)
        return self.ones_at_maxes(y_hat)
    
    def step(self, x):
        feed = {self.x : x, self.keep_prob: 1}
        fetch = [self.y_hat]
        for i in range(self.ncells):
            feed[self.layers[i]['istate'].c] = self.layers[i]['state_c']
            feed[self.layers[i]['istate'].h] = self.layers[i]['state_h']
            fetch.append(self.layers[i]['fstate'].c) ; fetch.append(self.layers[i]['fstate'].h)

        got = self.sess.run(fetch, feed)
        y_hat = got[0] ; states = got[1:]

        for i in range(self.ncells):
            self.layers[i]['state_c'] = states[2*i]
            self.layers[i]['state_h'] = states[2*i+1]

        return self.ones_at_maxes(y_hat)
    
    # if you are making a generative model, this function might be useful
    def generate(self, steps_forward, prev_x=None):
        assert self.xlen == self.ylen, "input and target dimensions should be the same"
        prev_x = np.zeros((1,1,self.xlen)) if prev_x is None else prev_x
        xs = np.zeros((1,0,self.ylen))

        for t in range(steps_forward):
            xs = np.concatenate((xs,prev_x), axis=1)
            prev_x = self.step(prev_x)
        return xs

    def ones_at_maxes(self, x):
        maxs = np.amax(x,axis=2, keepdims=True)
        t = np.tile(maxs,[1,1,x.shape[2]])
        ind = np.where(t==x)
        y = np.zeros_like(x) ; y[ind] = 1
        return y

    def count_params(self):
        # tf parameter overview
        total_parameters = 0 ; print("Model overview:")
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            print('\tvariable "{}" has {} parameters'.format(variable.name, variable_parameters) )
            total_parameters += variable_parameters
        print( "Total of {} parameters".format(total_parameters) )

    def try_load_model(self):
        # load saved model, if any
        global_step = 0
        self.saver = tf.train.Saver(tf.global_variables())
        load_was_success = True # yes, I'm being optimistic
        try:
            save_dir = '/'.join(self.save_path.split('/')[:-1])
            ckpt = tf.train.get_checkpoint_state(save_dir)
            load_path = ckpt.model_checkpoint_path
            self.saver.restore(self.sess, load_path)
        except:
            print( "no saved model to load. starting new session" )
            load_was_success = False
        else:
            print( "loaded model: {}".format(load_path) )
            self.saver = tf.train.Saver(tf.global_variables())
            global_step = int(load_path.split('-')[-1])
        return global_step

    def save(self, global_step):
        self.saver.save(self.sess, self.save_path, global_step=global_step)
