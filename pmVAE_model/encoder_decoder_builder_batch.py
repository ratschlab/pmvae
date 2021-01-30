import numpy as np
import tensorflow as tf
import sparse_layer

def encoder_builder(mask_encoder,activation_hidden="elu",*args, **kwargs):
    if len(mask_encoder)==1:
        encoder = tf.keras.Sequential()
        encoder.add(sparse_layer.SparseLayer(units=mask_encoder[0].shape[1],mask=mask_encoder[0],*args, **kwargs))
    else:
        encoder = tf.keras.Sequential()
        for i in range(len(mask_encoder)-1):
            if activation_hidden=="tanh":
                encoder.add(sparse_layer.SparseLayer(units=mask_encoder[i].shape[1],
                                                 mask=mask_encoder[i],activation=activation_hidden,use_bias=True,*args, **kwargs))
                encoder.add(tf.keras.layers.BatchNormalization())
            else:
                encoder.add(sparse_layer.SparseLayer(units=mask_encoder[i].shape[1],
                                                 mask=mask_encoder[i],activation="linear",use_bias=True,*args, **kwargs))
                encoder.add(tf.keras.layers.BatchNormalization())
                encoder.add(tf.keras.layers.Activation(activation_hidden))
                
        encoder.add(sparse_layer.SparseLayer(units=mask_encoder[-1].shape[1],use_bias=True,mask=mask_encoder[-1],activation="linear",*args, **kwargs))
        encoder.add(tf.keras.layers.BatchNormalization())
    return encoder

def decoder_builder(mask_decoder,activation_hidden="elu",bias_last_layer=True,*args, **kwargs):
    if len(mask_decoder)==1:
        decoder_1 = tf.keras.Sequential()
        decoder_1.add(sparse_layer.SparseLayer(mask=mask_decoder[0],units=mask_decoder[0].shape[1],*args, **kwargs))
        return decoder_1
        
    else:
        decoder_1 = tf.keras.Sequential()
        for i in range(len(mask_decoder)-1):
            if activation_hidden=="tanh":
                decoder_1.add(sparse_layer.SparseLayer(units=mask_decoder[i].shape[1],
                                                 mask=mask_decoder[i],use_bias=True,activation="tanh",*args, **kwargs))
                decoder_1.add(tf.keras.layers.BatchNormalization())
            else:
                decoder_1.add(sparse_layer.SparseLayer(units=mask_decoder[i].shape[1],
                                                 mask=mask_decoder[i],use_bias=True,activation="linear",*args, **kwargs))
                decoder_1.add(tf.keras.layers.BatchNormalization())
                decoder_1.add(tf.keras.layers.Activation(activation_hidden))

        decoder_2=sparse_layer.SparseLayer(units=mask_decoder[-1].shape[1],  mask=mask_decoder[-1],use_bias=bias_last_layer,activation="linear",*args, **kwargs)
        

        return decoder_1,decoder_2