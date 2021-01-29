import tensorflow as tf
import numpy as np
import pandas as pd
import time

def train_VAE(training_data,validation_data,batch_size,num_epochs,model,beta,optimizer,geneSets,
              verbose=False,auxillary_nodes=0,p_p_latent_dim=4,hidden=32,alpha=1,full_loss=False):
    """Trains the VAE, returns the trained model and the training history."""

    sub_mask=np.zeros((len(geneSets),len(geneSets)*hidden))
    for i in range(len(geneSets)):
        for j in range(hidden):
            sub_mask[i,j+((hidden*i))]=1
    sub_mask=sub_mask.astype(np.float32)

    
    @tf.function
    def aux_vae_loss(model, x):
        x_reconstructed, mu, log_var,z,last_layer = model.call(x)
        
        kl_div = tf.math.exp(log_var) + mu**2 - log_var - 1
        kl_div = 0.5 * tf.reduce_sum(kl_div, 1)
        reconstruction_loss_gl=tf.keras.losses.MSE(x,x_reconstructed)

        reconstruction_loss_lc=tf.constant(0.0)

        for i in range(len(sub_mask)-auxillary_nodes):
            mask=np.zeros((training_data.shape[1]))
            mask[geneSets[i]]=1
            z_path=tf.multiply(last_layer,sub_mask[i])
            x_reconstructed_m=model.merger(z_path)
            
            mask_1 = tf.cast(mask*x_reconstructed_m, dtype=tf.bool)
            
            model_out_masked=tf.boolean_mask(mask*x_reconstructed_m,mask_1)
            x_masked=tf.boolean_mask(x,mask_1)

            reconstruction_loss_lc=reconstruction_loss_lc+tf.keras.losses.MSE(x_masked,model_out_masked)
        
        reconstruction_loss_lc=reconstruction_loss_lc/(len(geneSets)-auxillary_nodes)
        reconstruction_loss=alpha*reconstruction_loss_lc+reconstruction_loss_gl
        
        return kl_div,reconstruction_loss,reconstruction_loss_lc,reconstruction_loss_gl


    @tf.function
    def compute_apply_gradients(model, x, optimizer,beta):
        """Computes the applies the gradients."""      
        with tf.GradientTape() as tape:
            kl_div,reconstruction_loss,reconstruction_loss_lc,reconstruction_loss_gl = aux_vae_loss(model, x)
            loss=reconstruction_loss+beta*kl_div

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return kl_div,reconstruction_loss,loss,reconstruction_loss_lc,reconstruction_loss_gl
        
                

    history=np.zeros((num_epochs,10))
    
    dataset= tf.data.Dataset.from_tensor_slices(training_data)
    dataset= dataset.shuffle(batch_size*5).batch(batch_size)
    
    num_batches= training_data.shape[0] // batch_size

    for epochs in range(num_epochs):
        tic = time.perf_counter()
       
        #Compute and apply gradients
        for step, x in enumerate(dataset):
            kl_div,reconstruction_loss,loss,reconstruction_loss_lc,reconstruction_loss_gl=compute_apply_gradients(model, x, optimizer,beta)
            loss_t=tf.reduce_mean(reconstruction_loss)+beta*tf.reduce_mean(kl_div)

        #Save training loss


        kl_div_val,reconstruction_loss_val,reconstruction_loss_lc_val,reconstruction_loss_gl_val = aux_vae_loss(model, validation_data)
        if full_loss==True:
            kl_div,reconstruction_loss,reconstruction_loss_lc,reconstruction_loss_gl = aux_vae_loss(model, training_data)
                
        loss_val=reconstruction_loss_val+beta*kl_div_val
        #Compute test loss

        history[epochs,0]=loss_t
        history[epochs,1]=tf.reduce_mean(reconstruction_loss)
        history[epochs,2]=tf.reduce_mean(kl_div)
        history[epochs,3]=tf.reduce_mean(loss_val)
        history[epochs,4]=tf.reduce_mean(reconstruction_loss_val)
        history[epochs,5]=tf.reduce_mean(kl_div_val)
        history[epochs,6]=tf.reduce_mean(reconstruction_loss_lc_val)
        history[epochs,7]=tf.reduce_mean(reconstruction_loss_gl_val)
        history[epochs,8]=tf.reduce_mean(reconstruction_loss_lc)
        history[epochs,9]=tf.reduce_mean(reconstruction_loss_gl)

        toc = time.perf_counter()
        
        if verbose:
            print(toc-tic)
            print("Epoch[{}/{}], Step[{}/{}],Loss: {:.4f}, Reconstruction loss: {:.4f}, KL div:{:.8f},val_loss:{:.4f},beta:{:.4f}"
                     .format(epochs+1,num_epochs,step+1,num_batches,loss_t, float(tf.reduce_mean(reconstruction_loss)),float(tf.reduce_mean(kl_div)),float(history[epochs,3]),beta))
    
    history=pd.DataFrame(history,columns=["Loss","Reconstruction loss","KL div","Val Loss","Val Reconstruction loss","Val KL div","Val Reconstruction loss lc", "Val Reconstruction loss gl", "Reconstruction loss lc", "Reconstruction loss gl"])
    
    return history,model