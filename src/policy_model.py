import tensorflow as tf
from ray.rllib.models.model import Model
from ray.rllib.models.misc import normc_initializer
from tensorflow.contrib import slim

class BalancedInputFC(Model):
    """
    Custom network which encodes higher-dimension position and time input
    before concatenation.
    """
    def _build_layers_v2(self, input_dict, num_outputs, options):
        """Define the layers of a custom model.

        Arguments:
            input_dict (dict): Dictionary of input tensors, including "obs",
                "prev_action", "prev_reward", "is_training".
            num_outputs (int): Output tensor must be of size
                [BATCH_SIZE, num_outputs].
            options (dict): Model options.

        Returns:
            (outputs, feature_layer): Tensors of size [BATCH_SIZE, num_outputs]
                and [BATCH_SIZE, desired_feature_size].
        """
        xy_encode = slim.fully_connected(input_dict["obs"][0], 256)
        t_encode = slim.fully_connected(input_dict["obs"][2], 128)
        fc1 = tf.concat([xy_encode, input_dict["obs"][1], t_encode], 1)
        fc2 = slim.fully_connected(fc1, 512)
        fc3 = slim.fully_connected(fc2, 256)
        print('Im hereeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee')
        #if 'constrain_outputs' in options:
        #constraints = tf.convert_to_tensor([0.01, 0.01, 0.01, 0.01])#options['constrain_outputs'])
        #outputs = slim.fully_connected(fc3, num_outputs, 
        #                                       weights_initializer=normc_initializer(0.01), 
        #                                       activation_fn=tf.nn.tanh)
    
        #outputs = tf.math.multiply(constraints, self.common_layer)
        
        #else:
        outputs = slim.fully_connected(fc3, num_outputs, 
                                       weights_initializer=normc_initializer(0.01),
                                       activation_fn=None)
            
        return outputs, fc3

    #def value_function(self):
    #    return tf.reshape(linear(self.common_layer, 1, "value", normc_initializer(1.0)), [-1])
