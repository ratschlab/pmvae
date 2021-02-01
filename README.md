# pmvae
pmVAE leverages biological prior information in the form of pathway gene sets to construct interpretable representations of single-cell data. It uses pathway module sub networks to construct a latent space factorized by pathway gene sets. Each pathway has a corresponding module which behaves as a mini VAE acting only on the participating genes. These modules produce latent representations that can be direcrly interpreted within the context of a specific pathway. To account for overlapping pathway gene sets (due to e.g. signaling hierarchies) we have a custom training procedure to encourage module independence.

More details can be found in our preprint: https://www.biorxiv.org/content/10.1101/2021.01.28.428664v1

The forward pass through modules are parallelized through the use of dense masking layers. These are similar to normal dense layers, except we multiply their kernels element-wise with a binary mask to remove unwanted connections. We use two types masks, one to assign genes to their modules and a block diagonal mask to remove connections between the module hidden layers.

repo is under construction, a fully runnable demo.ipynb will be available soon
