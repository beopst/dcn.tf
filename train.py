'''
Dynamic Capacity Networks (http://arxiv.org/abs/1511.07838) implementation using TensorFlow and slim library.

This code is using the cluttered MNITS dataset, which can be obtained in https://github.com/deepmind/mnist-cluttered.

Author: Sangheum Hwang
'''

from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
import time

from slim import slim
from models import dcn
import mnist_input
from random import shuffle

################### Parameters #######################
BATCH_SIZE = 64
NUM_EPOCHS = 100
NUM_EPOCHS_PER_DECAY = 50
INITIAL_LEARNING_RATE = 0.01
LEARNING_RATE_DECAY = 0.1
######################################################


def error_rate(predictions, labels):

    return 100.0 - (100.0 * np.sum(np.argmax(predictions,1)==labels) / 
                    predictions.shape[0])

def train():


    trn_data = mnist_input.load_data('data/mnist-cluttered/train')
    n_trn = len(trn_data)

    val_data = mnist_input.load_data('data/mnist-cluttered/valid', shuffle=False)
    val_data_x, val_data_y = zip(*val_data)
    n_val = len(val_data)

    step_counter = tf.Variable(0)
    num_batches_for_epoch = int(np.ceil(n_trn/BATCH_SIZE))
    decay_steps = int(num_batches_for_epoch * NUM_EPOCHS_PER_DECAY)

    trn_x = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 100, 100, 1))
    trn_y = tf.placeholder(tf.int32, shape=(BATCH_SIZE,))
    val_x = tf.placeholder(tf.float32, shape=(50, 100, 100, 1))

    logits = dcn.inference(trn_x)
    dcn.loss(logits,trn_y,batch_size=BATCH_SIZE)


    total_loss = tf.add_n(tf.get_collection(slim.losses.LOSSES_COLLECTION))
    batchnorm_updates = tf.get_collection(slim.ops.UPDATE_OPS_COLLECTION)
    batchnorm_updates_op = tf.group(*batchnorm_updates)

    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE, 
                                    step_counter, decay_steps, 
                                    LEARNING_RATE_DECAY, 
                                    staircase=True)
    #optimizer = tf.train.MomentumOptimizer(lr,0.9).minimize(total_loss,
    #                                                          global_step=step_counter)
    optimizer = tf.train.AdamOptimizer(lr).minimize(total_loss,
                                                    global_step=step_counter)
   
    train_op = tf.group(optimizer,batchnorm_updates_op)
    # create Session
    sess = tf.Session()
    tf.initialize_all_variables().run(session=sess)

    eval_logits = dcn.inference(val_x,is_training=False)
    eval_prediction = tf.nn.softmax(eval_logits)
    def _evaluate(data, sess):
        size = data.shape[0]
        predictions = np.ndarray(shape=(size,10),dtype=np.float32)
        for begin in xrange(0, size, 50):
            end = begin + 50
            feed_dict = {val_x: data[begin:end,...]}
            predictions[begin:end,:] = sess.run(eval_prediction, feed_dict=feed_dict)
        return predictions

    start_time = time.time()
    for step in xrange(NUM_EPOCHS*num_batches_for_epoch+1):

        if step % num_batches_for_epoch == 0:
            # shuffle training data
            shuffle(trn_data)
            trn_data_x, trn_data_y = zip(*trn_data)
       
            # validation
            predictions = _evaluate(np.asarray(val_data_x,dtype=np.float32),sess)
            eval_error = error_rate(predictions, np.asarray(val_data_y))
            print '\nValidation error: %.2f\n' % eval_error

        offset = (step * BATCH_SIZE) % (n_trn - BATCH_SIZE)
        batch_x = np.asarray(trn_data_x[offset:(offset+BATCH_SIZE)],dtype=np.float32)
        batch_y = np.asarray(trn_data_y[offset:(offset+BATCH_SIZE)],dtype=np.int32)

        _, res_lr, res_loss = sess.run([train_op,lr,total_loss], 
                                       feed_dict={trn_x:batch_x,trn_y:batch_y})

        if step % 100 == 0:
            elapsed_time = time.time() - start_time
            start_time = time.time()
            print 'Step %d (epoch %.2f), Elapsed: %.1f ms, LR: %.4f, Loss: %.4f' % \
                  (step, float(step)/num_batches_for_epoch, 1000*elapsed_time, res_lr, res_loss)


    import ipdb
    ipdb.set_trace()

def main(argv=None):
    train()

if __name__ == '__main__':
    tf.app.run()
