# StyleCariGAN in PyTorch

Official implementation of StyleCariGAN:Caricature Generation via StyleGAN Feature Map Modulation in PyTorch

## Requirements

* PyTorch 1.3.1
* CUDA 10.1/10.2

## Usage

First download pre-trained model weights:

> bash ./download.sh

### Test

> python test.py

This will generate caricatures from latent codes corresponding to photos in ./examples. You can see the results in ./results directory.

When you generate caricatures from your photos, you need to include latent code inversion:

> python test.py --invert_images

This will invert latent codes from input photos and generate caricatures from latent codes.
