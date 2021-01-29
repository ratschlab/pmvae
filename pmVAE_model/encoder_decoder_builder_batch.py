import numpy as np
import tensorflow as tf
import sparse_layer

def encoder_builder(mask_encoder,*args, **kwargs):
    if len(mask_encoder)==1:
        encoder = tf.keras.Sequential()
        encoder.add(sparse_layer.SparseLayer(units=mask_encoder[0].shape[1],mask=mask_encoder[0],*args, **kwargs))
    else:
        encoder = tf.keras.Sequential()
        for i in range(len(mask_encoder)-1):
            if kwargs['activation']=="tanh":
                encoder.add(sparse_layer.SparseLayer(units=mask_encoder[i].shape[1],
                                                 mask=mask_encoder[i],*args, **kwargs))
                encoder.add(tf.keras.layers.BatchNormalization())
            else:
                activation=kwargs['activation']
                kwargs['activation']="linear"
                encoder.add(sparse_layer.SparseLayer(units=mask_encoder[i].shape[1],
                                                 mask=mask_encoder[i],*args, **kwargs))
                encoder.add(tf.keras.layers.BatchNormalization())
                kwargs['activation']=activation
                encoder.add(tf.keras.layers.Activation(kwargs['activation']))
                


        kwargs['activation']=="linear"   
        encoder.add(sparse_layer.SparseLayer(units=mask_encoder[-1].shape[1],mask=mask_encoder[-1],*args, **kwargs))
        encoder.add(tf.keras.layers.BatchNormalization())
    return encoder

def decoder_builder(mask_decoder,bias_last_layer=True,*args, **kwargs):
    if len(mask_decoder)==1:
        decoder_1 = tf.keras.Sequential()
        decoder_1.add(sparse_layer.SparseLayer(mask=mask_decoder[0],units=mask_decoder[0].shape[1],*args, **kwargs))
        return decoder_1
        
    else:
        decoder_1 = tf.keras.Sequential()
        for i in range(len(mask_decoder)-1):
            if kwargs['activation']=="tanh":
                decoder_1.add(sparse_layer.SparseLayer(units=mask_decoder[i].shape[1],
                                                 mask=mask_decoder[i],*args, **kwargs))
                decoder_1.add(tf.keras.layers.BatchNormalization())
            else:
                activation=kwargs['activation']
                kwargs['activation']=="linear"
                decoder_1.add(sparse_layer.SparseLayer(units=mask_decoder[i].shape[1],
                                                 mask=mask_decoder[i],*args, **kwargs))
                decoder_1.add(tf.keras.layers.BatchNormalization())
                kwargs['activation']=activation
                decoder_1.add(tf.keras.layers.Activation(kwargs['activation']))
                
               
        kwargs['activation']="linear"
        if bias_last_layer==True:
            kwargs['use_bias']=True
        else:
            kwargs['use_bias']=False

        decoder_2=sparse_layer.SparseLayer(units=mask_decoder[-1].shape[1],
                                    mask=mask_decoder[-1],*args, **kwargs)
        

        return decoder_1,decoder_2