import tensorflow as tf

class VAE(tf.keras.Model):
    """Combines the encoder and decoder for training."""
    
    def __init__(self, dim,encoder,decoder_1,decoder_2, **kwargs):
        super(VAE, self).__init__(**kwargs)
        
        self.size_inp=dim[0]
        self.latent_dim = dim[1]
        self.decoder_1=decoder_1
        self.decoder_2=decoder_2
        self.encoder=encoder        
        
    def encode(self,x,**kwargs):
        mu, log_var = tf.split(self.encoder(x,**kwargs), num_or_size_splits=2, axis=1)
        return mu, log_var
    
    def decode(self,x,**kwargs):
        x_recon_1=self.decoder_1(x,**kwargs)
        x_reconstructed= self.decoder_2(x_recon_1,**kwargs)
        return x_reconstructed
    
    def merger(self,x):
        x_reconstructed= self.decoder_2(x)
        return x_reconstructed

    def reparametrize(self,mu,log_var):
        eps = tf.random.normal(log_var.shape)
        return mu + tf.math.exp(log_var / 2) * eps
    
    def call(self, x):
        mu, log_var= self.encode(x)
        z = self.reparametrize(mu,log_var)
        x_reconstructed=self.decode(z)
        last_layer=self.decoder_1(z)
        
        return x_reconstructed, mu, log_var,z,last_layer