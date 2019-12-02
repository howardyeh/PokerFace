import tensorflow as tf
from tensorflow.contrib.slim import nets
slim = tf.contrib.slim
    
        
class Model(object):
    def __init__(self, is_training):
        self._is_training = is_training
    
    def predict(self, preprocessed_inputs):
        # use ResNet_v1 as model
        with slim.arg_scope(nets.resnet_v1.resnet_arg_scope()):
            net, endpoints = nets.resnet_v1.resnet_v1_50(preprocessed_inputs, 
                                                        num_classes=None,
                                                        is_training=self._is_training)
        net = tf.squeeze(net, axis=[1, 2])

        # The output will be a single number indicating possibility
        logits = slim.fully_connected(net, num_outputs= 1,
                                      activation_fn=tf.nn.sigmoid, scope='Predict')

        return logits
    
    def loss(self, prediction, expectation, groundtruth):
        # The weighting between expectation and prediction need to be tuned
        logits = 0.5 * prediction + 0.5 * expectation
        slim.losses.sparse_softmax_cross_entropy(
            logits=logits, 
            labels=groundtruth_lists,
            scope='Loss')
        prob_loss = tf.nn.l2_loss(groundtruth - logits)
        slim.losses.add_loss(l2_loss)
        loss = slim.losses.get_total_loss()
        loss_dict = {'prob_loss': prob_loss,'loss': loss}
        return loss_dict

    def error_percent(self, prediction, groundtruth):
        error = abs(groundtruth - prediction)
        return error