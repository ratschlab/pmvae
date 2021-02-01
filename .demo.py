#!/usr/bin/env python
# coding: utf-8

# In[1]:


import anndata
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

from model import PMVAE
from train import train


# In[2]:


data = anndata.read(
    '/cluster/work/grlab/projects/projects2020-autoencoder_pathway/'
    'gamma_VAE/kang_2018/kang_count.h5ad')

membership_mask = data.varm['I'].astype(bool).T
trainset, testset = train_test_split(data.X, shuffle=True, test_size=0.25)

batch_size = 256
trainset = tf.data.Dataset.from_tensor_slices(trainset)
trainset = trainset.shuffle(5 * batch_size).batch(batch_size)


# In[3]:


model = PMVAE(
    membership_mask, 4, [12],
    beta=1e-05,
    kernel_initializer='he_uniform',
    bias_initializer='zero',
    activation='elu',
)

opt = tf.keras.optimizers.Adam(learning_rate=0.001)


# In[4]:


train(model, opt, trainset, testset, nepochs=1)

