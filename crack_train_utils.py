# Learning the Enigma with Recurrent Neural Networks
# Sam Greydanus. January 2017. MIT License.

'''Most of this file is identical to "train_utils." Minor modifications to the accuracy and sampling functions
were necessary. I could have kept this code in a single "train utils" file but 1) I may decide to modify the
other utils functions later and want to have that ability 2) like most undergraduate physics majors, I can't
resist writing ugly and redundant code :)'''

import time
import sys
import copy

import numpy as np
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# quantitative eval
def accuracy(model, data):
    sum_acc = 0 ; trials = 100
    kl = data.key_len
    for _ in range(trials):
        batch = data.next_batch(model.batch_size)
        y = np.stack(batch[1])
        y_hat = model.decode(batch[0])
        ignore = np.sum(y[:,-kl:,-1:])
        if y.shape[0]*kl != ignore:
            sum_acc += np.sum(y[:,-kl:,:-1]*y_hat[:,-kl:,:-1])/(y.shape[0]*kl - ignore) # TODO: make this less awk
    return sum_acc * 100 / trials

# qualitative look
def sample(model, data, FLAGS):
    model.reset_states()
    key = "KEY" ; message = "YOUKNOWNOTHINGJONSNOW"[:data.wordlen]
    plaintext = message + '-'*(data.key_len)
    # key = unicode(key, "utf-8") ; plaintext = unicode(plaintext, "utf-8") # need this for enigma only
    
    ciphertext = data.encode(key, message) + '-'*(data.key_len)
    decoded = ''
    for i in range(len(ciphertext)):
        p1 = data.one_hot(ciphertext[i])
        p2 = data.one_hot(plaintext[i])
        p = np.concatenate([p1, p2], axis=1)
        p = np.tile(p,[1,1,1])

        c = model.step(p)
        ix = np.where(c[0,0,:] == np.amax(c[0,0,:]))
        ix = np.squeeze(ix[0])
        if ix <= 25: # TODO: make this less awk
            decoded += data.A[ix]
        else:
            decoded += '-'
    
    print( "\t\tplaintext is:  '{}'".format( message + '-'*(data.key_len) ) )
    print( "\t\tciphertext is: '{}'".format(ciphertext) )
    print( "\t\tkey is:        '{}'".format('-'*(len(message) + data.key_len - len(key)) + key) )
    print( "\t\tprediction is: '{}'".format(decoded) )
    return plaintext, decoded

def train(model, data, log, FLAGS):
    running_loss = None # for smoothing the loss over time
    start = time.time()

    # Restore train state. There's DEFINITELY a more efficient way to do this but I don't want to spend my
    #   time dealing with tedious file I/O. Still gonna send it! https://www.youtube.com/watch?v=WIrWyr3HgXI
    global_step = model.try_load_model()
    if global_step == 0:
        log.clear_fs() ; print('\tresetting log files...')
    else:
        loss_hist = log.read('loss')
        acc_hist = log.read('acc')
        log.clear_fs()
        for i in range(loss_hist.shape[0]):
            if loss_hist[i,0] <= global_step:
                log.write_loss('{},{},{}'.format(loss_hist[i,0], loss_hist[i,1],loss_hist[i,2]))
        for i in range(acc_hist.shape[0]):
            if acc_hist[i,0] <= global_step:
                log.write_acc('{},{},{},{},{}'\
                    .format(acc_hist[i,0], acc_hist[i,1], acc_hist[i,2], acc_hist[i,3], acc_hist[i,4]))

    print('training...')
    while global_step < FLAGS.total_steps:
        global_step += 1
        
        data_start = time.time()
        batch = data.next_batch(model.batch_size)

        rnn_start = time.time()
        train_loss = model.train_step(batch)

        # bookkeeping starts here
        data_time = rnn_start-data_start ; rnn_time = time.time() - rnn_start ; tot_time = time.time() - data_start
        running_loss = train_loss if running_loss is None else 0.99*running_loss + 0.01*train_loss
        log.write_loss('{},{},{}'.format(global_step, train_loss*100, tot_time))

        if global_step%FLAGS.print_every == 0:
            print( "\tstep: {:7d} | loss: {:.3f} | data time: {:.4f} sec | rnn time: {:.4f} sec | total time: {:.4f} sec"\
                .format(global_step, running_loss*100, data_time, rnn_time, tot_time) )
        if global_step%FLAGS.acc_every == 0:
            train_acc = accuracy(model,data)
            print('\t\taccuracy: {}'.format(train_acc))
            sample(model, data, FLAGS)
            log.write_acc('{},{},{},{},{}'.format(global_step, train_acc, data_time, rnn_time, tot_time))
        if global_step%FLAGS.save_every == 0 and global_step!=0:
            model.save(global_step)
            print( "\t\t...SAVED MODEL #{}".format(global_step) )

class Logger():
    def __init__(self, FLAGS):
        self.lossf = FLAGS.meta_dir + 'loss_hist.txt'
        self.accf = FLAGS.meta_dir + 'acc_hist.txt'
        
    def clear_fs(self):
        with open(self.accf, 'w') as f: f.write('')
        with open(self.lossf, 'w') as f: f.write('')
            
    def write_acc(self, s):
        with open(self.accf, 'a') as f:
            f.write(s + "\n")
            
    def write_loss(self, s):
        with open(self.lossf, 'a') as f:
            f.write(s + "\n")
            
    def read(self, mode='loss'):
        file = None
        if mode is 'loss':
            file = self.lossf
        elif mode is 'acc':
            file = self.accf
        else:
            print('error: mode not recognized')
        hist = []
        with open(file, 'r') as f:
            for line in f:
                data = [float(s) for s in line.split(',')]
                hist.append(data)
        return np.vstack(hist)

def plot_stats(log, FLAGS):
    loss_hist = log.read('loss')
    acc_hist = log.read('acc')
    f1 = plt.figure(figsize=[16,5])

    plt.subplot(121)
    plt.plot(loss_hist[:,0], loss_hist[:,1], linewidth=3.0, label='loss')
    plt.title('Loss', fontsize=14)
    plt.xlabel('train step', fontsize=14) ; plt.setp(plt.gca().axes.get_xticklabels(), fontsize=14)
    plt.ylabel('loss', fontsize=14) ; plt.setp(plt.gca().axes.get_yticklabels(), fontsize=14)
    plt.ylim([0,45])

    plt.subplot(122)
    plt.plot(acc_hist[:,0], acc_hist[:,1], linewidth=3.0, label='accuracy')
    plt.title('Test accuracy', fontsize=14)
    plt.xlabel('train step', fontsize=14) ; plt.setp(plt.gca().axes.get_xticklabels(), fontsize=14)
    plt.ylabel('accuracy (%)', fontsize=14) ; plt.setp(plt.gca().axes.get_yticklabels(), fontsize=14)

    # estimate total time required to do this run
    tt_sec = np.sum(loss_hist[:,2])
    m, s = divmod(tt_sec, 60)
    h, m = divmod(m, 60)
    tt_pretty = "%02dh %02dm %02ds"%(h, m, s)
    results_msg = '\nlearning rate : {}\nbatch size: {}\ntrain time: {}\nfinal accuracy: {:.2f}%'\
        .format(FLAGS.lr, FLAGS.batch_size, tt_pretty, acc_hist[-1,1])
    print('\nTRAIN SUMMARY:' + results_msg + '\n')
    f1.text(0.92, .50, results_msg, ha='left', va='center', fontsize=12)
    plt.ylim([0,100])

    title = "RNN on {} task".format(FLAGS.cipher[0].upper() + FLAGS.cipher[1:])
    f1.text(0.4, .97, title, ha='left', va='center', fontsize=18)

    f1.savefig('./{}train-{}.png'.format(FLAGS.meta_dir, FLAGS.cipher), bbox_inches='tight')