
from slim import ops
from slim import scopes
from slim import variables
from slim import losses

import tensorflow as tf

######## Parameters ########
N_PATCHES = 8
############################

def top_layers(inputs):

    out = ops.conv2d(inputs, 96, [4,4], stride=2, scope='top_conv1')
    _,fm_size,fm_size,_ = out.get_shape()
    out = ops.max_pool(out, [fm_size,fm_size], stride=1, scope='top_gpool')

    out = ops.flatten(out, scope='top_flatten')
    out = ops.fc(out, 10, activation=None, bias=0.0, scope='top_logits')

    return out

def coarse_layers(inputs):

    out = ops.conv2d(inputs, 12, [7,7], stride=2, padding='VALID', scope='coarse_conv1')
    out = ops.conv2d(out, 24, [3,3], stride=2, padding='VALID', scope='coarse_conv2')

    return out

def fine_layers(inputs):

    out = ops.conv2d(inputs, 24, [3,3], stride=1, padding='VALID', scope='fine_conv1')
    out = ops.conv2d(out, 24, [3,3], stride=1, padding='VALID', scope='fine_conv2')
    out = tf.pad(out, [[0,0],[1,1],[1,1],[0,0]])

    out = ops.max_pool(out, [2,2], stride=2, scope='fine_pool1')
    
    out = ops.conv2d(out, 24, [3,3], stride=1, padding='VALID', scope='fine_conv3')
    out = tf.pad(out, [[0,0],[1,1],[1,1],[0,0]])
    out = ops.conv2d(out, 24, [3,3], stride=1, padding='VALID', scope='fine_conv4')
    out = tf.pad(out, [[0,0],[1,1],[1,1],[0,0]])

    out = ops.max_pool(out, [2,2], stride=2, scope='fine_pool2')

    out = ops.conv2d(out, 24, [3,3], stride=1, padding='VALID', scope='fine_conv5')

    return out

def entropy(coarse_logits):
    """Calculate the entropy of the coarse model output
    """
    return -tf.reduce_sum(coarse_logits*tf.log(coarse_logits))

def identify_saliency(grads):
    """Identify top k saliency scores.

       Args.
            grads: gradient of the entropy wrt features
       Trick.
            use tf.nn.top_k ops to extract position indices
    """

    M = tf.sqrt(tf.reduce_sum(tf.mul(grads,grads),3))
    top_k_values, top_k_idxs = tf.nn.top_k(ops.flatten(M), N_PATCHES)

    return top_k_values, top_k_idxs, M

def extract_features(inputs, k_idxs, map_h):
    """Extract top k fine features

       Trick.
            use tf.image.extract_glimpse ops to get input patches
    """

    def _extract_feature(inputs, idxs):

        idxs = tf.expand_dims(idxs,1)

        idx_i = tf.floordiv(idxs, map_h)
        idx_j = tf.mod(idxs, map_h)

        # NOTE: 
        # calculate the center of input batches
        # this depends on coarse layer's architecture
        origin_i = 2*(2*idx_i+1)+3
        origin_j = 2*(2*idx_j+1)+3

        origin_centers = tf.concat(1,[origin_i,origin_j])
        origin_centers = tf.to_float(origin_centers)

        # NOTE: size also depends on the architecture
        patches = tf.image.extract_glimpse(inputs, size=[14,14], offsets=origin_centers, 
                                           centered=False, normalized=False)

        fine_features = fine_layers(patches)

        # reuse variables
        tf.get_variable_scope().reuse_variables()
        
        src_idxs = tf.concat(1,[idx_i,idx_j])

        return fine_features, src_idxs

    k_features = []
    k_src_idxs = []
    for i in xrange(N_PATCHES):
        fine_feature, src_idx = _extract_feature(inputs,k_idxs[:,i])
        k_features.append(fine_feature)
        k_src_idxs.append(src_idx)

    return k_features, k_src_idxs



def replace_features(coarse_features, fine_features, replace_idxs):
    """ Replace fine features with the corresponding coarse features

        Trick.
            use tf.dynamic_stitch ops
    """
    
    def _convert_to_1d_idxs(src_idxs):
        """ Convert 2D idxs to 1D idxs 
            within 1D tensor whose shape is (b*h*w*c)
        """

        batch_1d = map_channel.value * map_width.value * src_idxs[:,0] + \
                   map_channel.value * src_idxs[:,1]
        flat_idxs = [batch_1d+i for i in xrange(map_channel.value)]
        flat_idxs = tf.reshape(tf.transpose(tf.pack(flat_idxs)), [-1])

        return flat_idxs

    batch_size, map_height, map_width, map_channel = coarse_features.get_shape()

    # flatten coarse features
    flat_coarse_features = tf.reshape(coarse_features, [batch_size.value,-1])
    flat_coarse_features = tf.reshape(flat_coarse_features, [-1])

    # flatten fine features
    flat_fine_features = [tf.reshape(i,[-1]) for i in fine_features]
    flat_fine_features = tf.concat(0,flat_fine_features)

    flat_fine_idxs = [_convert_to_1d_idxs(i) for i in replace_idxs]
    flat_fine_idxs = tf.concat(0,flat_fine_idxs)

    # extract coarse features to be replaced
    # this is required for hint-based training
    flat_coarse_replaced = tf.stop_gradient(tf.gather(flat_coarse_features, flat_fine_idxs))

    merged = tf.dynamic_stitch([tf.range(0,flat_coarse_features.get_shape()[0]),flat_fine_idxs],
            [flat_coarse_features,flat_fine_features])

    merged = tf.reshape(merged,coarse_features.get_shape())

    return merged, flat_coarse_replaced, flat_fine_features

def inference(inputs, is_training=True, scope=''):

    batch_norm_params = {'decay': 0.9, 'epsilon': 0.001}

    with scopes.arg_scope([ops.conv2d, ops.fc], weight_decay=0.0,
                          is_training=is_training, batch_norm_params=batch_norm_params):
        # get features from coarse layers
        coarse_features = coarse_layers(inputs)
        coarse_features_dim = coarse_features.get_shape()[1] # width

        # calculate saliency scores and extract top k
        coarse_output = top_layers(coarse_features)
        coarse_h = entropy(tf.nn.softmax(coarse_output))
        coarse_grads = tf.gradients(coarse_h, coarse_features)
        top_k_values, top_k_idxs, M = identify_saliency(coarse_grads[0])

        # get features from fine layers
        fine_features, src_idxs = extract_features(inputs, top_k_idxs, coarse_features_dim)

        # merge two feature maps
        merged, flat_coarse, flat_fine = replace_features(coarse_features, fine_features, src_idxs)

        # add additional L2 norm to LOSSES_COLLECTION
        #losses.l2_loss(flat_coarse - tf.stop_gradient(flat_fine), weight=0.001, scope='objective_hint')
       
        final_logits = top_layers(merged)

    return final_logits

def loss(logits, labels, batch_size):

    sparse_labels = tf.reshape(labels, [batch_size,1])
    indices = tf.reshape(tf.range(batch_size), [batch_size,1])
    concated = tf.concat(1, [indices, sparse_labels])
    num_classes = logits.get_shape()[-1].value
    dense_labels = tf.sparse_to_dense(concated, [batch_size, num_classes], 1.0, 0.0)

    losses.cross_entropy_loss(logits, dense_labels, label_smoothing=0.0, weight=1.0)

    


