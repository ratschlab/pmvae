import numpy as np
import tensorflow as tf

from .utils import (build_encoder_net,
                    build_decoder_net,
                    build_module_isolation_mask)


from collections import namedtuple
Outputs = namedtuple('Outputs', 'z global_recon module_outputs mu logvar')


class PMVAE(tf.keras.Model):
    """Combines the encoder and decoder for training."""

    def __init__(self,
                 membership_mask,
                 module_latent_dim,
                 hidden_layers,
                 bias_last_layer=False,
                 add_auxiliary_module=False,
                 beta=1,
                 **kwargs):
        '''pmVAE: factorizes latent space by pathways

        membership_mask: mask assigning features (genes) to modules
        module_latent_dim: dimension of each module latent space
        hidden_layers: width of each module encoder/decoder hidden layer
        '''

        super(PMVAE, self).__init__()
        self.num_annotated_modules, self.num_feats = membership_mask.shape
        self.add_auxiliary_module = add_auxiliary_module
        if add_auxiliary_module:
            membership_mask = np.vstack(
                    (membership_mask, np.ones_like(membership_mask[0])))

        self.beta = beta

        self.encoder_net = build_encoder_net(
                membership_mask,
                hidden_layers,
                module_latent_dim,
                **kwargs)

        self.decoder_net, self.merge_layer = build_decoder_net(
                membership_mask,
                hidden_layers,
                module_latent_dim,
                bias_last_layer=bias_last_layer,
                **kwargs)

        self.membership_mask = membership_mask
        self.module_isolation_mask = build_module_isolation_mask(
                self.membership_mask.shape[0],
                hidden_layers[-1])
        return

    def encode(self, x, **kwargs):
        params = self.encoder_net(x, **kwargs)
        mu, logvar = tf.split(params, num_or_size_splits=2, axis=1)
        return mu, logvar

    def decode(self, z, **kwargs):
        module_outputs = self.decoder_net(z, **kwargs)
        global_recon = self.merge(module_outputs, **kwargs)
        return global_recon

    def merge(self, module_outputs, **kwargs):
        global_recon = self.merge_layer(module_outputs, **kwargs)
        return global_recon

    def reparametrize(self, mu, logvar):
        eps = tf.random.normal(logvar.shape)
        return mu + tf.math.exp(logvar / 2) * eps

    def call(self, x, **kwargs):
        mu, logvar = self.encode(x, **kwargs)
        z = self.reparametrize(mu, logvar)
        module_outputs = self.decoder_net(z, **kwargs)
        global_recon = self.merge(module_outputs, **kwargs)
        outputs = Outputs(z, global_recon, module_outputs, mu, logvar)
        return outputs

    def get_masks_for_local_losses(self):
        if self.add_auxiliary_module:
            return zip(self.membership_mask[:-1],
                       self.module_isolation_mask[:-1])

        return zip(self.membership_mask,
                   self.module_isolation_mask)
