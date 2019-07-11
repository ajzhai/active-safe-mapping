import tensorflow as tf
from ray.rllib.models import Model
from tensorflow.contrib import slim

class BalancedInputFC(Model):
    """
    Custom network which encodes higher-dimension position and time input
    befor concatenation.
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
        fc1 = tf.concat([xy_encode, input_dict["obs"][1], t_encode], 0)
        fc2 = slim.fully_connected(fc1, 512)
        fc3 = slim.fully_connected(fc2, 256)
        if 'constrain_outputs' in options:
            constraints = tf.convert_to_tensor(options['constrain_outputs'])
            outputs = slim.fully_connected(fc3, num_outputs, activation_fn=tf.nn.tanh)
            outputs = tf.math.multiply(constraints, outputs)
        else:
            outputs = slim.fully_connected(fc3, num_outputs, activation_fn=None)
        return outputs, fc3
