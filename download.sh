mkdir -p data
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-N7wPpYUf_QcG5566WVZlaxVC90M7NNE' -O ./data/kang_count.h5ad

mkdir -p data/published
wget 'https://public.bmi.inf.ethz.ch/projects/2020/pmvae/kang_recons.h5ad' -O ./data/published/kang_recons.h5ad
