# Decoding the Autokey Cipher with Recurrent Neural Networks
# Sam Greydanus. January 2017. MIT License.

import time
import sys
import copy

import numpy as np
import tensorflow as tf

# quantitative eval
def accuracy(model, dataloader):
    mean_acc = 0 ; trials = 100
    mkl = dataloader.max_key_len
    for _ in range(trials):
        batch = dataloader.next_batch(model.batch_size)
        y = np.stack(batch[1])
        y_hat = model.decode(batch[0])
        mean_acc += 1./trials * np.sum(y[:,mkl:,:]*y_hat[:,mkl:,:])/(y.shape[0]*(y.shape[1]-mkl)) # TODO: make this less awk
    return mean_acc*100

# qualitative eval
def sample(model, dataloader):
    model.reset_states()
    plaintext = 'YOUKNOWNOTHINGJONSNOW'
    key = dataloader.rands(np.random.randint(dataloader.max_key_len-1) + 1)
    ciphertext = key + '-'*(dataloader.max_key_len - len(key)) + dataloader.encode(key, plaintext)
    decoded = ''
    for i in range(len(ciphertext)):
        p = np.tile(dataloader.one_hot(ciphertext[i]),[1,1,1])
        c = model.step(p)
        ix = np.where(c[0,0,:] == np.amax(c[0,0,:]))
        ix = np.squeeze(ix[0])
        if ix <= 25: # TODO: make this less awk
            decoded += dataloader.A[ix]
        else:
            decoded += '-'
    
    print "plaintext is:  '{}'".format(key + '-'*(dataloader.max_key_len - len(key)) + plaintext)
    print "ciphertext is: '{}'".format(ciphertext)
    print "prediction is: '{}'".format(decoded)
    return plaintext, decoded

# train loop
def train(model, dataloader, FLAGS):
    replay = [] # replay-like dataset for reusing synthesized batches
    running_loss = None # for smoothing the loss over time
    start = time.time()
    global_step = model.try_load_model()
    while True:
        global_step += 1
        if global_step%FLAGS.turnover==0 or len(replay) < FLAGS.replay_queue_depth:
            batch = dataloader.next_batch(model.batch_size)
            replay.append(batch)
        else:
            batch = replay[np.random.randint(len(replay))]
            if len(replay) > FLAGS.replay_queue_depth: replay = replay[1:]
        train_loss = model.train_step(batch)
        running_loss = train_loss if running_loss is None else 0.99*running_loss + 0.01*train_loss
        
        if global_step%FLAGS.print_every == 0:
            print "\tstep {}, loss {:3f}, batch time {:3f}".format(global_step, running_loss*100, (time.time()-start)/100)
            start = time.time()
        if global_step%FLAGS.save_every == 0 and global_step!=0:
            model.save(global_step)
            print "SAVED MODEL #{}".format(global_step)
            print "ACCURACY: {:3f}%".format(accuracy(model,dataloader))
