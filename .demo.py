#!/usr/bin/env python
# coding: utf-8

# In[49]:


import anndata
import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
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


# In[10]:


model = PMVAE(
    membership_mask, 4, [12],
    add_auxiliary_module=True,
    beta=1e-05,
    kernel_initializer='he_uniform',
    bias_initializer='zero',
    activation='elu',
)

opt = tf.keras.optimizers.Adam(learning_rate=0.001)


# In[4]:


history = train(model, opt, trainset, testset, nepochs=1200)


# In[11]:


outputs = model(data.X)


# In[31]:




def embeddings_to_df(codes, terms, index, add_auxiliary=True):
    terms = list(terms)
    if add_auxiliary:
        terms.append('AUXILIARY')
    terms = pd.Series(terms)
    
    latent_dim_per_pathway = codes.shape[1] // terms.size
    term_index = np.tile(range(latent_dim_per_pathway), terms.size).astype(str)
    terms = terms.repeat(latent_dim_per_pathway) + '-' + term_index.astype(str)
    
    return pd.DataFrame(codes, columns=terms.values, index=index)


# In[48]:



outdir = Path('./results')
outdir.mkdir(exist_ok=True, parents=True)

recons = anndata.AnnData(
    outputs.global_recon.numpy(),
    obs=data.obs,
    uns=data.uns,
    varm=data.varm,
)

recons.obsm['codes'] = embeddings_to_df(
    outputs.mu.numpy(),
    data.uns['terms'],
    data.obs_names)

recons.obsm['logvar'] = embeddings_to_df(
    outputs.logvar.numpy(),
    data.uns['terms'],
    data.obs_names)

data.write(outdir/'recons.h5ad')

