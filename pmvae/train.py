import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm

from collections import namedtuple
Loss = namedtuple('Loss', 'loss kl recon global_recon local_recon')


def train(model, opt, trainset, testset, nepochs, debug=False):

    @tf.function
    def pmvae_loss(model, inputs):

        def weighted_mse(y_true, y_pred, sample_weight):
            sample_weight = tf.convert_to_tensor(sample_weight, dtype=y_pred.dtype)
            diff = tf.square(y_true - y_pred) * sample_weight
            wmse = tf.reduce_sum(diff, -1) / tf.reduce_sum(sample_weight)
            return wmse

        outputs = model(inputs)
        kl = tf.math.exp(outputs.logvar) + outputs.mu**2 - outputs.logvar - 1
        kl = 0.5 * tf.reduce_sum(kl, 1)

        global_recon_loss = tf.keras.losses.MSE(inputs, outputs.global_recon)

        local_recon_loss = tf.constant(0.0)
        for feat_mask, module_mask in model.get_masks_for_local_losses():
            # dropout other modules & reconstruct
            only_active_module = tf.multiply(outputs.module_outputs, module_mask)
            local_recon = model.merge(only_active_module)

            # only compute the loss with participating genes
            wmse = weighted_mse(inputs, local_recon, feat_mask)

            local_recon_loss = local_recon_loss + wmse

        local_recon_loss = local_recon_loss / model.num_annotated_modules


        loss = Loss(
                loss=global_recon_loss + local_recon_loss + model.beta * kl,
                recon=global_recon_loss + local_recon_loss,
                global_recon=global_recon_loss,
                local_recon=local_recon_loss,
                kl=kl
                )

        return loss

    @tf.function
    def compute_apply_gradients(model, x, opt):
        """Computes the applies the gradients."""
        with tf.GradientTape() as tape:
            loss = pmvae_loss(model, x)

        gradients = tape.gradient(loss.loss, model.trainable_variables)
        opt.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    fields = [
        'train-loss', 'train-recon', 'train-kl', 'train-local', 'train-global',
        'test-loss', 'test-recon', 'test-kl', 'test-local', 'test-global']
    history = pd.DataFrame(np.nan, index=np.arange(nepochs), columns=fields)

    for epoch in tqdm(range(nepochs)):

        # Compute and apply gradients
        for step, x in enumerate(trainset):
            train_loss = compute_apply_gradients(model, x, opt)

        test_loss = pmvae_loss(model, testset)

        history.loc[epoch, 'train-loss'] = train_loss.loss.numpy().mean()
        history.loc[epoch, 'train-kl'] = train_loss.kl.numpy().mean()
        history.loc[epoch, 'train-recon'] = train_loss.recon.numpy().mean()
        history.loc[epoch, 'train-local'] = train_loss.local_recon.numpy().mean()
        history.loc[epoch, 'train-global'] = train_loss.global_recon.numpy().mean()

        history.loc[epoch, 'test-loss'] = test_loss.loss.numpy().mean()
        history.loc[epoch, 'test-kl'] = test_loss.kl.numpy().mean()
        history.loc[epoch, 'test-recon'] = test_loss.recon.numpy().mean()
        history.loc[epoch, 'test-local'] = test_loss.local_recon.numpy().mean()
        history.loc[epoch, 'test-global'] = test_loss.global_recon.numpy().mean()

    return history
