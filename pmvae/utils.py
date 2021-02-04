import numpy as np
import tensorflow as tf
from scipy.linalg import block_diag
from tensorflow.python.keras.layers import ops
from tensorflow.keras.layers import Dense, Activation, BatchNormalization


def build_pathway_mask(nfeats, membership_mask, hidden_layers):
    '''Connects genes to pathway modules

    Repeats the membership mask for each module input node
    See M in Methods 2.2
    '''
    return np.repeat(membership_mask, hidden_layers, axis=0).T


def build_separation_mask(input_dim, out_put_dim, nmodules):
    '''Removes connections betweens pathway modules

    Block diagonal matrix, see Sigma in Methods 2.2
    '''
    blocks = [np.ones((input_dim, out_put_dim))] * nmodules
    return block_diag(*blocks)


def build_module_isolation_mask(nmodules, module_output_dim):
    '''Isolates a single module for gradient steps

    Used for the local reconstruciton terms, drops all modules except one
    '''
    blocks = [np.ones((1, module_output_dim))] * nmodules
    return block_diag(*blocks)


def build_base_masks(membership_mask, hidden_layers, latent_dim):
    '''Builds the masks used by encoders/decoders

    membership_mask: boolean array, modules x features
    hidden_layers: width of each hidden layer (list of ints)
    latent_dim: size of each module latent dim

    pathway mask assigns genes to pathway modules
    separation masks keep modules separated
    Encoder modifies the last separation mask to give mu/logvar
    Decoder reverses and transposes the masks
    '''
    nmodules, nfeats = membership_mask.shape
    base = list()
    base.append(build_pathway_mask(nfeats, membership_mask, hidden_layers[0]))
    dims = hidden_layers + [latent_dim]
    for dinput, doutput in zip(dims[:-1], dims[1:]):
        base.append(build_separation_mask(dinput, doutput, nmodules))

    base = [mask.astype(np.float32) for mask in base]
    return base


def build_encoder_net(
        membership_mask,
        hidden_layers,
        latent_dim,
        activation='elu',
        batch_norm=True,
        **kwargs):

    masks = build_base_masks(membership_mask, hidden_layers, latent_dim)
    masks[-1] = np.hstack((masks[-1], masks[-1]))

    encoder_net = tf.keras.Sequential()
    for mask in masks[:-1]:
        encoder_net.add(MaskedLayer(mask.shape[1], mask, **kwargs))
        if batch_norm:
            encoder_net.add(BatchNormalization())
        encoder_net.add(Activation(activation))

    encoder_net.add(MaskedLayer(masks[-1].shape[1], masks[-1],  **kwargs))
    if batch_norm:
        encoder_net.add(BatchNormalization())

    return encoder_net


def build_decoder_net(
        membership_mask,
        hidden_layers,
        latent_dim,
        activation='elu',
        batch_norm=True,
        bias_last_layer=False,
        **kwargs):
    masks = build_base_masks(membership_mask, hidden_layers, latent_dim)
    masks = [mask.T for mask in masks[::-1]]

    decoder_net = tf.keras.Sequential()
    for mask in masks[:-1]:
        decoder_net.add(MaskedLayer(mask.shape[1], mask, **kwargs))
        if batch_norm:
            decoder_net.add(BatchNormalization())
        decoder_net.add(Activation(activation))

    kwargs.pop('use_bias', None)
    merge_layer = MaskedLayer(masks[-1].shape[1], masks[-1], use_bias=bias_last_layer, **kwargs)
    return decoder_net, merge_layer


class MaskedLayer(Dense):
    def __init__(self, units, mask, *args, **kwargs):
        super(MaskedLayer, self).__init__(units, *args, **kwargs)
        self.mask = tf.Variable(mask, trainable=False)
        return

    def build(self, *args, **kwargs):
        super(MaskedLayer, self).build(*args, **kwargs)
        assert self.mask.shape == self.kernel.shape
        return

    def call(self, inputs):
        return ops.core.dense(
            inputs,
            tf.multiply(self.kernel, self.mask),
            self.bias,
            self.activation,
            dtype=self._compute_dtype_object)
