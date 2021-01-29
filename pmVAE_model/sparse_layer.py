import tensorflow as tf
from tensorflow.python.keras.layers import ops

class SparseLayer(tf.keras.layers.Dense):
    def __init__(self, units, mask, *args, **kwargs):
        super(SparseLayer, self).__init__(units, *args, **kwargs)
        self.mask = tf.Variable(mask, trainable=False)

    def build(self, *args, **kwargs):
        super(SparseLayer, self).build(*args, **kwargs)
        assert self.mask.shape == self.kernel.shape
        return

    def call(self, inputs):
        return ops.core.dense(
            inputs,
            tf.multiply(self.kernel, self.mask),
            self.bias,
            self.activation,
            dtype=self._compute_dtype_object)