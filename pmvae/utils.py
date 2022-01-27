import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.linalg import block_diag
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
    merge_layer = MaskedLayer(
        masks[-1].shape[1], masks[-1],
        use_bias=bias_last_layer,
        **kwargs)

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
        if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            inputs = tf.cast(inputs, dtype=self._compute_dtype_object)

        kernel = tf.multiply(self.kernel, self.mask)
        outputs = tf.matmul(a=inputs, b=kernel)

        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs


def load_annotations(gmt, genes, min_genes=10):
    genesets = parse_gmt(gmt, genes, min_genes)
    annotations = pd.DataFrame(False, index=genes, columns=genesets.keys())
    for key, genes in genesets.items():
        annotations.loc[genes, key] = True

    return annotations


def parse_gmt(path, symbols=None, min_genes=10):
    lut = dict()
    for line in open(path, 'r'):
        key, _, *genes = line.strip().split()
        if symbols is not None:
            genes = symbols.intersection(genes).tolist()
        if len(genes) < min_genes:
            continue
        lut[key] = genes

    return lut


def make_annotation_mask(membership, genes, min_genes=15):
    if isinstance(genes, pd.Series):
        index = genes.index
    else:
        genes = pd.Index(genes)
        index = genes

    mask = pd.DataFrame(False, index=index, columns=membership.keys())
    for key, mems in membership.items():
        if isinstance(genes, pd.Series):
            mems = index[genes.isin(mems)]
        else:
            mems = genes.intersection(mems)

        mask.loc[mems, key] = True

    mask = mask.drop(columns=mask.columns[mask.sum(0) < min_genes])

    return mask


def add_annotation(
            data,
            membership=None, gmt=None,
            names=None,
            key='annotation',
            **kwargs):
    '''Add memberships to varm of data

    data: AnnData instance
    gmt: path to gmt memberships
    memberships: dict mapping gene set names to genes
    names: gene symbols (or var key) of symbols matching membership vals
    key: varm key to add annotations
    kwargs: make_annotaiton_mask kwargs
    '''

    if names is None:
        names = data.var_names
    elif isinstance(names, str):
        names = data.var[names]

    if gmt is not None:
        assert membership is None
        membership = parse_gmt(gmt)

    mask = make_annotation_mask(membership, names, **kwargs)
    data.varm[key] = mask

    if 'annotated' not in data.var:
        data.var['annotated'] = False
    data.var['annotated'] = data.var['annotated'] | (data.varm[key].sum(1) > 0)

    return
