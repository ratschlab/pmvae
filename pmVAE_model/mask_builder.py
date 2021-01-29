import numpy as np

def pathway_mask_builder(input_dim,geneSets,hidden_layers):
    pathway_mask=np.zeros((input_dim,len(geneSets)*hidden_layers)).astype(np.float32)
    counter=0
    for i in range(len(geneSets)):
        for j in range(hidden_layers):
            pathway_mask[geneSets[i].astype(int),counter]=1
            counter=counter+1
    return pathway_mask

def separation_mask_builder(input_dim,out_put_dim,geneSets):
    input_size=len(geneSets)*input_dim
    output_size=len(geneSets)*out_put_dim
    separation_mask=np.zeros((input_size,output_size)).astype(np.float32)
    for i in range(len(geneSets)):
        for l in range(input_dim):
            for m in range(out_put_dim):
                separation_mask[l+(input_dim*i),m+(out_put_dim*i)]=1
    return separation_mask

def mask_builder(input_dim,hidden_layers,geneSets,p_p_latent_dim):
    mask=[]
    if len(hidden_layers)==0:
        mask.append(pathway_mask_builder(input_dim,geneSets,hidden_layers=p_p_latent_dim))
    else:
        mask.append(pathway_mask_builder(input_dim,geneSets,hidden_layers[0]))
        if len(hidden_layers)>1:
            for i in range(len(hidden_layers)-1):
                mask.append(separation_mask_builder(input_dim=hidden_layers[i],out_put_dim=hidden_layers[i+1],geneSets=geneSets))
        out=separation_mask_builder(input_dim=hidden_layers[-1],out_put_dim=p_p_latent_dim,geneSets=geneSets)
        mask.append(out)
    
    mask_decoder=[]
    for i in reversed(range(len(mask))):
        mask_decoder.append(np.transpose(mask[i]))
    mask[-1]=np.hstack((mask[-1],mask[-1]))
    
    mask_encoder=mask
    
    return mask_encoder,mask_decoder

