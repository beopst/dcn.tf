
from slim import ops
from slim import scopes
from slim import variables
from slim import losses

import tensorflow as tf


def top_layers(inputs):

    with tf.variable_scope('top_layers'):
        out = ops.conv2d(inputs, 96, [4,4], stride=2, padding='VALID', scope='top_conv1')
        _,fm_size,fm_size,_ = out.get_shape()
        out = ops.max_pool(out, [fm_size,fm_size], stride=1, scope='top_gpool')

        out = ops.flatten(out, scope='top_flatten')
        #out = ops.fc(out, 10, activation=None, bias=0.0, batch_norm_params=None, scope='top_logits')
        bn_out = {'decay': 0.99, 'epsilon': 0.001, 'scale':True}
        out = ops.fc(out, 10, activation=None, batch_norm_params=bn_out, scope='top_logits')

    return out

def coarse_layers(inputs):

    with tf.variable_scope('coarse_layers'):
        out = ops.conv2d(inputs, 12, [7,7], stride=2, padding='VALID', scope='coarse_conv1')
        out = ops.conv2d(out, 24, [3,3], stride=2, padding='VALID', scope='coarse_conv2')

    return out

def inference(inputs, is_training=True, scope=''):

    if not is_training:
        tf.get_variable_scope().reuse_variables()

    batch_norm_params = {'decay': 0.99, 'epsilon': 0.001}

    with scopes.arg_scope([ops.conv2d, ops.fc], weight_decay=0.0005,
                          is_training=is_training, batch_norm_params=batch_norm_params):
        
        coarse_features = coarse_layers(inputs)
        final_logits = top_layers(coarse_features)

    return final_logits, tf.constant(0.0)

def loss(logits, labels, batch_size):

    sparse_labels = tf.reshape(labels, [batch_size,1])
    indices = tf.reshape(tf.range(batch_size), [batch_size,1])
    concated = tf.concat(1, [indices, sparse_labels])
    num_classes = logits.get_shape()[-1].value
    dense_labels = tf.sparse_to_dense(concated, [batch_size, num_classes], 1.0, 0.0)

    losses.cross_entropy_loss(logits, dense_labels, label_smoothing=0.0, weight=1.0)

    


