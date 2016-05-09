'''
Dynamic Capacity Networks (http://arxiv.org/abs/1511.07838) implementation using TensorFlow and slim library.

This code is using the cluttered MNIST dataset, which can be obtained in https://github.com/deepmind/mnist-cluttered.

Author: Sangheum Hwang
'''

from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
import time

from slim import slim
from models import dcn as model
import mnist_input
from random import shuffle

################### Parameters #######################
BATCH_SIZE = 64
NUM_EPOCHS = 300
NUM_EPOCHS_PER_DECAY = 100
INITIAL_LEARNING_RATE = 0.001
LEARNING_RATE_DECAY = 0.1
HINT_WEIGHT = 0.01
GPU = 3
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

    step_counter = tf.Variable(0, trainable=False)
    num_batches_for_epoch = int(np.ceil(n_trn/BATCH_SIZE))
    decay_steps = int(num_batches_for_epoch * NUM_EPOCHS_PER_DECAY)

    with tf.device('/gpu:%d'%GPU):

        trn_x = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 100, 100, 1))
        trn_y = tf.placeholder(tf.int32, shape=(BATCH_SIZE,))
        val_x = tf.placeholder(tf.float32, shape=(50, 100, 100, 1))

        logits, hint_loss = model.inference(trn_x)
        model.loss(logits,trn_y,batch_size=BATCH_SIZE)

        # regularization losses are in slim.losses.LOSSES_COLLECTION collection
        total_loss = tf.add_n(tf.get_collection(slim.losses.LOSSES_COLLECTION))
        
        batchnorm_updates = tf.get_collection(slim.ops.UPDATE_OPS_COLLECTION)
        batchnorm_updates_op = tf.group(*batchnorm_updates)

        lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE, 
                                        step_counter, decay_steps, 
                                        LEARNING_RATE_DECAY, 
                                        staircase=True)
        #optimizer = tf.train.MomentumOptimizer(lr,0.9).minimize(total_loss,
        #                                                          global_step=step_counter)
        optimizer = tf.train.AdamOptimizer(lr)

        top_vars = slim.variables.get_variables('top_layers')
        fine_vars = slim.variables.get_variables('fine_layers')
        #fine_vars = []
        coarse_vars = slim.variables.get_variables('coarse_layers')
        #coarse_vars = []

        optimizer = tf.train.AdamOptimizer(lr)
        top_grads = optimizer.compute_gradients(total_loss, 
          var_list=top_vars)
        fine_grads = optimizer.compute_gradients(total_loss, 
          var_list=fine_vars)
        #fine_grads = []
        coarse_grads = optimizer.compute_gradients(total_loss + hint_loss * HINT_WEIGHT, 
          var_list=coarse_vars)
        #coarse_grads = []

        apply_gradients = optimizer.apply_gradients(top_grads + fine_grads +  coarse_grads, global_step=step_counter)
        train_op = tf.group(apply_gradients, batchnorm_updates_op)

        eval_logits, eval_hint_loss = model.inference(val_x,is_training=False)
        eval_prediction = tf.nn.softmax(eval_logits)
    
    # create Session
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    summary_writer = tf.train.SummaryWriter('./trials',sess.graph)
    tf.initialize_all_variables().run(session=sess)

    def _evaluate(data, sess):
        size = data.shape[0]
        predictions = np.ndarray(shape=(size,10),dtype=np.float32)
        for begin in xrange(0, size, 50):
            end = begin + 50
            feed_dict = {val_x: data[begin:end,...]}
            predictions[begin:end,:] = sess.run(eval_prediction, feed_dict=feed_dict)
        return predictions

    elapsed_trn_time = 0
    
    for step in xrange(NUM_EPOCHS*num_batches_for_epoch+1):

        if step % num_batches_for_epoch == 0:
            # shuffle training data
            shuffle(trn_data)
            trn_data_x, trn_data_y = zip(*trn_data)
       
            # validation
            start_time = time.time()
            predictions = _evaluate(np.asarray(val_data_x,dtype=np.float32),sess)
            eval_error = error_rate(predictions, np.asarray(val_data_y))
            elapsed_val_time = time.time() - start_time

            print '\n[Validation] Epoch: %.2f, Elapsed: %.1f ms, Error: %.2f\n' \
                  % (float(step)/num_batches_for_epoch, 1000*elapsed_val_time, eval_error)

        offset = (step * BATCH_SIZE) % (n_trn - BATCH_SIZE)
        batch_x = np.asarray(trn_data_x[offset:(offset+BATCH_SIZE)],dtype=np.float32)
        batch_y = np.asarray(trn_data_y[offset:(offset+BATCH_SIZE)],dtype=np.int32)

        start_time = time.time()
        _, res_lr, res_loss, res_hint, = sess.run([train_op,lr,total_loss, hint_loss], 
                                       feed_dict={trn_x:batch_x,trn_y:batch_y})
        #res_patches = sess.run(k_patches, 
        #                               feed_dict={trn_x:batch_x,trn_y:batch_y})
        elapsed_trn_time += time.time() - start_time

        if step % 100 == 0:
            print 'Step %d (epoch %.2f), Elapsed: %.1f ms, LR: %.5f, Loss: %.4f Hint loss: %.4f' % \
                  (step, float(step)/num_batches_for_epoch, 1000*elapsed_trn_time, res_lr, res_loss, res_hint)
            elapsed_trn_time = 0



    #import ipdb
    #ipdb.set_trace()

def main(argv=None):
    train()

if __name__ == '__main__':
    tf.app.run()
