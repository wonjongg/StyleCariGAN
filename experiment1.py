import os
import numpy as np
import torch
from torch import nn
from torchvision import utils
from invert import *
from exaggeration_model import StyleCariGAN
from align import ImageAlign
 
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else 'cpu'

    ### Experiment1
    ### generate z-code with gaussian ditribution
    ### and project to w-code with StyleCariGAN's photo generator
    
    ## constant
    samples = 1
    latent_dim = 512
    size = 256
    n_mlp = 8
    
    ## generate z-code
    sample_z = torch.randn(samples, 512, device = device)
    # print(sample_z.size())

    ## bring the checkpoint of StyleCariGAN
    ckpt = "checkpoint/StyleCariGAN/001000.pt"
    ckpt = torch.load(ckpt)

    ## project to w-code
    g_ema = StyleCariGAN(size, latent_dim, n_mlp).to(device)
    g_ema.load_state_dict(ckpt['g_ema'],strict = False)
    w  = g_ema.photo_generator.style(sample_z)

    '''
    Photo Generation
    '''
    utils.save_image(g_ema.photo_generator([sample_z])[0], os.path.join("experiment1_result",f'result_exp1b_photo.png'),range=(-1, 1),normalize= True)
    # print(w)
    # print(w.shape)

    ## import styles
    predefined_style = "style_palette/style_palette.npy"
    styles = torch.from_numpy(np.load(predefined_style)).to(device)
    styles = styles.unsqueeze(1)
    styles_new = styles[1].repeat(samples,14, 1)
    exaggeration_factor = 1.0
    truncation = 0.7
    truncation_latent = 1

    #styles_new = []
    #styles_new.append(truncation_latent + truncation *(styles[0]-truncation_latent))
    #print(styles_new.shape)
    phi = [1-exaggeration_factor]*4

    # print(styles_new.shape)
    
    ## StyleCariGAN
    imga = g_ema([w],[styles[45]], truncation = truncation, input_is_latent=True, truncation_latent= truncation_latent,mode= 'p2c', phi = phi)
    utils.save_image(imga['result'], os.path.join("experiment1_result",f'result_exp1a.png'),range=(-1, 1))