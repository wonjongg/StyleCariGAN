import os
import math
import numpy as np
import argparse

import torch
from torchvision import transforms, utils

from PIL import Image
from tqdm import tqdm

from exaggeration_model import StyleCariGAN


class perceptual_module(torch.nn.Module):
    def __init__(self):
        import torchvision
        super().__init__()
        perceptual = torchvision.models.vgg16(pretrained=True)
        self.module1_1 = torch.nn.Sequential(
            *list(perceptual.children())[0][:1])
        self.module1_2 = torch.nn.Sequential(
            *list(perceptual.children())[0][1:3])
        self.module3_2 = torch.nn.Sequential(
            *list(perceptual.children())[0][3:13])
        self.module4_2 = torch.nn.Sequential(
            *list(perceptual.children())[0][13:20])

    def forward(self, x):
        outputs = {}
        out = self.module1_1(x)
        outputs['1_1'] = out
        out = self.module1_2(out)
        outputs['1_2'] = out
        out = self.module3_2(out)
        outputs['3_2'] = out
        out = self.module4_2(out)
        outputs['4_2'] = out

        return outputs


# re-normalize image into vgg image normalization scheme
class TO_VGG(object):
    def __init__(self, device="cuda"):
        self.s_mean = torch.from_numpy(np.asarray([0.5, 0.5, 0.5])).view(
            1, 3, 1, 1).type(torch.FloatTensor).to(device)
        self.s_std = torch.from_numpy(np.asarray([0.5, 0.5, 0.5])).view(
            1, 3, 1, 1).type(torch.FloatTensor).to(device)
        self.t_mean = torch.from_numpy(np.asarray([0.485, 0.456, 0.406])).view(
            1, 3, 1, 1).type(torch.FloatTensor).to(device)
        self.t_std = torch.from_numpy(np.asarray([0.229, 0.224, 0.225])).view(
            1, 3, 1, 1).type(torch.FloatTensor).to(device)

    def __call__(self, t):
        t = (t + 1) / 2
        t = (t - self.t_mean) / self.t_std
        # t = t * self.t_std + self.t_mean
        return t


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, device):
    return make_noise(batch, latent_dim, 1, device)


def noise_regularize(noises):
    loss = 0

    for noise in noises:
        size = noise.shape[2]

        while True:
            loss = (
                loss +
                (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2) +
                (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2))

            if size <= 8:
                break

            noise = noise.reshape([-1, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([3, 5])
            size //= 2

    return loss


def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()

        noise.data.add_(-mean).div_(std)


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    return make_noise(batch, latent_dim, 1, device)


def l2loss(input1, input2):
    diff = input1 - input2
    diff = diff.pow(2).mean().sqrt().squeeze()
    return diff


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def invert(g_ema, perceptual, real_img, device, args):
    save = args.save
    result = {}
    to_vgg = TO_VGG()
    requires_grad(perceptual, True)
    requires_grad(g_ema, True)
    log_size = int(math.log(256, 2))
    num_layers = (log_size - 2) * 2 + 1

    w = args.mean_w.clone().detach().to(device).unsqueeze(0)  # 1 * 512
    w.requires_grad = True
    wplr = args.wlr
    optimizer = torch.optim.Adam(
        [w],
        lr=args.wlr,
    )

    with torch.no_grad():
        sample, _ = g_ema([w], input_is_latent=True, randomize_noise=True)
        if save:
            utils.save_image(
                sample,
                args.result_dir + f"/{args.image_name}_recon_w_initial.png",
                nrow=int(sample.shape[0]**0.5),
                normalize=True,
                range=(-1, 1),
            )
            utils.save_image(
                real_img,
                args.result_dir + f"/{args.image_name}_input.png",
                nrow=int(real_img.shape[0]**0.5),
                normalize=True,
                range=(-1, 1),
            )

    print('optimizing w')

    # loop for w
    pbar = range(args.w_iterations)
    pbar = tqdm(pbar, initial=0, dynamic_ncols=True, smoothing=0.01)
    for idx in pbar:
        if idx + 1 % (args.w_iterations // 2) == 0:
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] * args.lr_decay_rate
                wplr = wplr * args.lr_decay_rate

        real_img_vgg = to_vgg(real_img)
        t = 1
        w_tilde = w + torch.randn(w.shape, device=device) * t * t
        fake_img, _ = g_ema([w_tilde],
                            input_is_latent=True,
                            randomize_noise=True)
        fake_img_vgg = to_vgg(fake_img)

        fake_feature = perceptual(fake_img_vgg)
        real_feature = perceptual(real_img_vgg)

        loss_pixel = l2loss(fake_img, real_img)

        loss_feature = []
        for (fake_feat, real_feat) in zip(fake_feature.values(),
                                          real_feature.values()):
            loss_feature.append(l2loss(fake_feat, real_feat))
        loss_feature = torch.mean(torch.stack(loss_feature))

        loss = args.lambda_l2 * loss_pixel +\
                args.lambda_p * loss_feature

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description((
            f"optimizing w: loss_pixel: {loss_pixel:.4f}; loss_feature: {loss_feature:.4f}"
        ))

        if idx % (args.w_iterations // 3) == 0 and save:
            with torch.no_grad():
                sample, _ = g_ema([w],
                                  input_is_latent=True,
                                  randomize_noise=True)
                utils.save_image(
                    sample,
                    args.result_dir + f"/{args.image_name}_recon_w_{idx}.png",
                    nrow=int(sample.shape[0]**0.5),
                    normalize=True,
                    range=(-1, 1),
                )

    result['w'] = w.squeeze().cpu()

    if save:
        with torch.no_grad():
            sample, _ = g_ema([w], input_is_latent=True, randomize_noise=True)
            utils.save_image(
                sample,
                args.result_dir +
                f"/{args.image_name}_recon_w_final.png",  # should be same as recon_w_final.png
                nrow=int(sample.shape[0]**0.5),
                normalize=True,
                range=(-1, 1),
            )

    print('optimizing wp')

    # starting point for w : mean w
    wp = w.unsqueeze(1).repeat(1, args.num_layers,
                               1).detach().clone()  # single image
    wp.requires_grad = True

    noises = []
    for layer_idx in range(num_layers):
        res = (layer_idx + 5) // 2
        shape = [1, 1, 2**res, 2**res]
        noises.append(torch.randn(*shape, device=device).normal_())
        noises[layer_idx].requires_grad = True

    optimizer = torch.optim.Adam(
        [wp] + noises,
        lr=wplr,
    )

    if save:
        with torch.no_grad():
            sample, _ = g_ema(wp,
                              noise=noises,
                              input_is_w_plus=True,
                              randomize_noise=False)
            utils.save_image(
                sample,
                args.result_dir +
                f"/{args.image_name}_recon_wp_initial.png",  # should be same as recon_w_final.png
                nrow=int(sample.shape[0]**0.5),
                normalize=True,
                range=(-1, 1),
            )

    # loop for wp
    pbar = range(args.wp_iterations)
    pbar = tqdm(pbar, initial=0, dynamic_ncols=True, smoothing=0.01)
    for idx in pbar:
        if idx + 1 % (args.wp_iterations // 6) == 0:
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] * args.lr_decay_rate
        real_img_vgg = to_vgg(real_img)
        # loss
        t = max(1 - 3 * idx / args.wp_iterations, 0)
        wp_tilde = wp + torch.randn(wp.shape, device=device) * t * t
        fake_img, _ = g_ema(wp_tilde,
                            noise=noises,
                            input_is_w_plus=True,
                            randomize_noise=False)
        fake_img_vgg = to_vgg(fake_img)

        fake_feature = perceptual(fake_img_vgg)
        real_feature = perceptual(real_img_vgg)

        loss_pixel = l2loss(fake_img, real_img)

        loss_feature = []
        for (fake_feat, real_feat) in zip(fake_feature.values(),
                                          real_feature.values()):
            loss_feature.append(l2loss(fake_feat, real_feat))
        loss_feature = torch.mean(torch.stack(loss_feature))

        loss_noise = noise_regularize(noises)

        loss = args.lambda_l2 * loss_pixel +\
                args.lambda_p * loss_feature +\
                args.lambda_noise * loss_noise

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        noise_normalize_(noises)

        # update pbar
        pbar.set_description((
            f"optimizing wp: loss_pixel: {loss_pixel:.4f}; loss_feature: {loss_feature:.4f}"
        ))

        # save progress
        if idx % (args.wp_iterations // 3) == 0 and save:
            with torch.no_grad():
                sample, _ = g_ema(wp,
                                  noise=noises,
                                  input_is_w_plus=True,
                                  randomize_noise=False)
                utils.save_image(
                    sample,
                    args.result_dir + f"/{args.image_name}_recon_wp_{idx}.png",
                    nrow=int(sample.shape[0]**0.5),
                    normalize=True,
                    range=(-1, 1),
                )
    # end of optimization - save results
    with torch.no_grad():
        fake_img, _ = g_ema(wp,
                            noise=noises,
                            input_is_w_plus=True,
                            randomize_noise=False)
    if save:
        utils.save_image(
            fake_img,
            args.result_dir + f"/{args.image_name}_recon_final.png",
            nrow=int(fake_img.shape[0]**0.5),
            normalize=True,
            range=(-1, 1),
        )
    result['wp'] = wp.squeeze().cpu()
    result['noise'] = [n.cpu() for n in noises]
    torch.save(result, args.result_dir + f'/{args.image_name}.pt')


def parse_args():
    parser = argparse.ArgumentParser(description="StyleGAN2 encoder test")
    parser.add_argument("--w_iterations", type=int, default=250)
    parser.add_argument("--wp_iterations", type=int, default=2000)
    parser.add_argument('--image',
                        type=str,
                        required=True,
                        help='path to the image to invert')
    parser.add_argument("--size",
                        type=int,
                        default=256,
                        help="image sizes for the model")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--lambda_l2", type=float, default=1)
    parser.add_argument("--lambda_p", type=float, default=1)
    parser.add_argument("--lambda_noise", type=float, default=1e5)
    parser.add_argument("--wlr", type=float, default=4e-3)
    parser.add_argument("--lr_decay_rate", type=float, default=0.2)
    parser.add_argument("--result_dir", type=str, default='./invert_result')
    args = parser.parse_args()
    args.save = False  # no need to save optimized image
    return args


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    args = parse_args()

    args.latent = 512
    args.num_layers = 14

    g_ema = StyleCariGAN(args.size, args.latent, 8,
                         channel_multiplier=2).to(device)

    checkpoint = torch.load(args.ckpt)
    g_ema.load_state_dict(checkpoint['g_ema'], strict=False)
    g_ema = g_ema.photo_generator
    g_ema.eval()

    perceptual = perceptual_module().to(device)
    perceptual.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
    ])

    n = 50000
    samples = 256
    w = []
    for _ in range(n // samples):
        sample_z = mixing_noise(samples, args.latent, 0, device=device)
        w.append(g_ema.style(sample_z))
    w = torch.cat(w, dim=0)
    args.mean_w = w.mean(dim=0)

    os.makedirs(args.result_dir, exist_ok=True)
    photo = transform(Image.open(
        args.image).convert('RGB')).unsqueeze(0).to(device)
    args.image_name = args.image.split('/')[-1].split('.')[0]
    wp = invert(g_ema, perceptual, photo, device, args)
