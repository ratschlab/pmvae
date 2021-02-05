# pmvae
<p align="center">
  <img src="https://github.com/ratschlab/pmvae/blob/main/model.png" height="350">

pmVAE leverages biological prior information in the form of pathway gene sets to construct interpretable representations of single-cell data. It uses pathway module sub networks to construct a latent space factorized by pathway gene sets. Each pathway has a corresponding module which behaves as a mini VAE acting only on the participating genes. These modules produce latent representations that can be direcrly interpreted within the context of a specific pathway. To account for overlapping pathway gene sets (due to e.g. signaling hierarchies) we have a custom training procedure to encourage module independence.

More details can be found in our preprint: https://www.biorxiv.org/content/10.1101/2021.01.28.428664v1

To optimize speed on GPUs, the forward pass through modules are parallelized through the use of dense masking layers. These are similar to normal dense layers, except we multiply their kernels element-wise with a binary mask to remove unwanted connections. We use two types masks, one to assign genes to their modules and a block diagonal mask to remove connections between the module hidden layers. Using GPU (GeForce GTX 1080 Ti) a training epoch on the kang dataset (~10k cells, ~1k genes, ~100 pathways) takes around 3 seconds.

We would like to thank Rybakov et al. [1] for pointing us to the Kang et al. [2] work as well as for hosting their preprocessing.

[1] Rybakov, Sergei, Mohammad Lotfollahi, Fabian J. Theis, and F. Alexander Wolf. 2020. “Learning Interpretable Latent Autoencoder Representations with Annotations of Feature Sets.” Cold Spring Harbor Laboratory. https://doi.org/10.1101/2020.12.02.401182.

[2] Kang, Hyun Min, Meena Subramaniam, Sasha Targ, Michelle Nguyen, Lenka Maliskova, Elizabeth McCarthy, Eunice Wan, et al. 2018. “Multiplexed Droplet Single-Cell RNA-Sequencing Using Natural Genetic Variation.” Nature Biotechnology 36 (1): 89–94.
