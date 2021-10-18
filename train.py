import argparse
import math
import os
import random

import numpy as np
import torch
import torch.distributed as dist
from torch import autograd, nn, optim
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms, utils
from tqdm import tqdm

from facenet_pytorch import InceptionResnetV1

from distributed import (get_rank, get_world_size, reduce_loss_dict,
                         reduce_sum, synchronize)
from exaggeration_model import StyleCariGAN
from model import Discriminator, Discriminator_feat, ResNet18


# override requires_grad function
def requires_grad(model, flag=True, target_layer=None):
    for name, param in model.named_parameters():
        if target_layer is None:  # every layer
            param.requires_grad = flag
        elif target_layer in name:  # target layer
            param.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(outputs=real_pred.sum(),
                               inputs=real_img,
                               create_graph=True)
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0],
                                            -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3])
    grad, = autograd.grad(outputs=(fake_img * noise).sum(),
                          inputs=latents,
                          create_graph=True)
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() -
                                            mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None


def train(args, generator, discriminator_photo, discriminator_cari,
          discriminator_feat_p, discriminator_feat_c, g_optim, d_optim_p,
          d_optim_c, d_optim_fp, d_optim_fc, g_ema, p_cls, c_cls, id_net,
          device):

    pbar = range(args.iter)

    if get_rank() == 0:
        if not os.path.exists(f'checkpoint/{args.name}'):
            os.makedirs(f'checkpoint/{args.name}')

        if not os.path.exists(f'sample/{args.name}'):
            os.makedirs(f'sample/{args.name}')

        pbar = tqdm(pbar,
                    initial=args.start_iter,
                    dynamic_ncols=True,
                    smoothing=0.01)

    d_loss_val = 0
    d_feat_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    gan_loss_val = 0
    gan_feat_loss_val = 0
    idt_loss_val = 0
    attr_loss_val = 0
    loss_dict = {}

    if args.distributed:
        g_module = generator.module
        d_module_p = discriminator_photo.module
        d_module_c = discriminator_cari.module
        d_module_feat_p = discriminator_feat_p.module
        d_module_feat_c = discriminator_feat_c.module

    else:
        g_module = generator
        d_module_p = discriminator_photo
        d_module_c = discriminator_cari
        d_module_feat_p = discriminator_feat_p
        d_module_feat_c = discriminator_feat_c

    accum = 0.5**(32 / (10 * 1000))

    sample_z = torch.randn(args.n_sample, args.latent, device=device)

    criterion_BCE = nn.BCELoss()

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")
            break
        '''
        Discriminator for feat cari
        '''
        requires_grad(generator, False)
        requires_grad(discriminator_feat_c, True)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        noise_fine = mixing_noise(args.batch, args.latent, args.mixing, device)

        ret = generator(noise,
                        noise_fine,
                        truncation_latent=mean_latent,
                        mode='p2c')
        fake_feat = ret['co']
        real_feat = ret['gt_co'].detach()

        fake_pred = discriminator_feat_c(fake_feat)
        real_pred = discriminator_feat_c(real_feat)
        d_loss = d_logistic_loss(real_pred, fake_pred)

        loss_dict["d_feat_c"] = d_loss
        loss_dict["real_score_feat_c"] = real_pred.mean()
        loss_dict["fake_score_feat_c"] = fake_pred.mean()

        discriminator_feat_c.zero_grad()
        d_loss.backward()
        d_optim_fc.step()

        d_regularize = i % args.d_reg_every == 0

        if d_regularize:
            real_feat.requires_grad = True
            real_pred = discriminator_feat_c(real_feat)
            r1_loss = d_r1_loss(real_pred, real_feat)

            discriminator_feat_c.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every +
             0 * real_pred[0]).backward()

            d_optim_fc.step()

        '''
        Discriminator for feat photo
        '''
        requires_grad(generator, False)
        requires_grad(discriminator_feat_p, True)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        noise_fine = mixing_noise(args.batch, args.latent, args.mixing, device)

        ret = generator(noise,
                        noise_fine,
                        truncation_latent=mean_latent,
                        mode='c2p')
        fake_feat = ret['po']
        real_feat = ret['gt_po'].detach()

        fake_pred = discriminator_feat_p(fake_feat)
        real_pred = discriminator_feat_p(real_feat)
        d_loss = d_logistic_loss(real_pred, fake_pred)

        loss_dict["d_feat_p"] = d_loss
        loss_dict["real_score_feat_p"] = real_pred.mean()
        loss_dict["fake_score_feat_p"] = fake_pred.mean()

        discriminator_feat_p.zero_grad()
        d_loss.backward()
        d_optim_fp.step()

        d_regularize = i % args.d_reg_every == 0

        if d_regularize:
            real_feat.requires_grad = True
            real_pred = discriminator_feat_p(real_feat)
            r1_loss = d_r1_loss(real_pred, real_feat)

            discriminator_feat_p.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every +
             0 * real_pred[0]).backward()

            d_optim_fp.step()

        '''
        Discriminator for cari
        '''
        requires_grad(generator, False)
        requires_grad(discriminator_cari, True)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        noise_fine = mixing_noise(args.batch, args.latent, args.mixing, device)

        fake_img = generator(noise,
                             noise_fine,
                             truncation_latent=mean_latent,
                             mode='p2c')['result']
        real_img = generator(noise,
                             noise_fine,
                             truncation_latent=mean_latent,
                             mode='c_gt')
        real_img = real_img.detach()

        fake_pred = discriminator_cari(fake_img)
        real_pred = discriminator_cari(real_img)
        d_loss = d_logistic_loss(real_pred, fake_pred)

        loss_dict["d_c"] = d_loss
        loss_dict["real_score_c"] = real_pred.mean()
        loss_dict["fake_score_c"] = fake_pred.mean()

        discriminator_cari.zero_grad()
        d_loss.backward()
        d_optim_c.step()

        d_regularize = i % args.d_reg_every == 0

        if d_regularize:
            real_img.requires_grad = True
            real_pred = discriminator_cari(real_img)
            r1_loss = d_r1_loss(real_pred, real_img)

            discriminator_cari.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every +
             0 * real_pred[0]).backward()

            d_optim_c.step()

        '''
        Discriminator for photo
        '''
        requires_grad(generator, False)
        requires_grad(discriminator_photo, True)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        noise_fine = mixing_noise(args.batch, args.latent, args.mixing, device)

        fake_img = generator(noise,
                             noise_fine,
                             truncation_latent=mean_latent,
                             mode='c2p')['result']
        real_img = generator(noise,
                             noise_fine,
                             truncation_latent=mean_latent,
                             mode='p_gt')
        real_img = real_img.detach()

        fake_pred = discriminator_photo(fake_img)
        real_pred = discriminator_photo(real_img)
        d_loss = d_logistic_loss(real_pred, fake_pred)

        loss_dict["d_p"] = d_loss
        loss_dict["real_score_p"] = real_pred.mean()
        loss_dict["fake_score_p"] = fake_pred.mean()

        discriminator_photo.zero_grad()
        d_loss.backward()
        d_optim_p.step()

        d_regularize = i % args.d_reg_every == 0

        if d_regularize:
            real_img.requires_grad = True
            real_pred = discriminator_photo(real_img)
            r1_loss = d_r1_loss(real_pred, real_img)

            discriminator_photo.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every +
             0 * real_pred[0]).backward()

            d_optim_p.step()

        loss_dict["r1"] = r1_loss

        if args.distributed:
            requires_grad(generator.module.deformation_blocks_CP, True)
            requires_grad(generator.module.deformation_blocks_PC, True)
        else:
            requires_grad(generator.deformation_blocks_CP, True)
            requires_grad(generator.deformation_blocks_PC, True)


        requires_grad(discriminator_photo, False)
        requires_grad(discriminator_cari, False)

        requires_grad(discriminator_feat_p, False)
        requires_grad(discriminator_feat_c, False)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        ret_p2c = generator(noise, truncation_latent=mean_latent, mode='p2c')
        ret_p2c_recon = generator(noise,
                                  truncation_latent=mean_latent,
                                  mode='p2c_recon')
        ret_c2p = generator(noise, truncation_latent=mean_latent, mode='c2p')

        cyc_loss_p2c = 0
        cyc_loss_c2p = 0
        for lv in range(len(ret_p2c['po'])):
            cyc_loss_p2c += F.mse_loss(ret_p2c['po'][lv], ret_p2c['ro'][lv])
            cyc_loss_c2p += F.mse_loss(ret_c2p['co'][lv], ret_c2p['ro'][lv])

        cyc_loss = (cyc_loss_p2c + cyc_loss_c2p) / 2

        attr_p_p2c = p_cls(ret_p2c['org']).detach()
        attr_c_p2c = c_cls(ret_p2c['result'])

        attr_loss_p2c = criterion_BCE(attr_c_p2c, attr_p_p2c)

        attr_c_c2p = c_cls(ret_c2p['org']).detach()
        attr_p_c2p = p_cls(ret_c2p['result'])

        attr_loss_c2p = criterion_BCE(attr_p_c2p, attr_c_c2p)

        attr_loss = (attr_loss_p2c + attr_loss_c2p) / 2

        fake_pred_photo = discriminator_photo(ret_c2p['result'])
        fake_pred_cari = discriminator_cari(ret_p2c['result'])

        gan_loss_p2c = g_nonsaturating_loss(fake_pred_cari)
        gan_loss_c2p = g_nonsaturating_loss(fake_pred_photo)

        gan_loss = (gan_loss_p2c + gan_loss_c2p) / 2

        fake_feat_photo = discriminator_feat_p(ret_c2p['po'])
        fake_feat_cari = discriminator_feat_c(ret_p2c['co'])

        gan_feat_loss_p2c = g_nonsaturating_loss(fake_feat_cari)
        gan_feat_loss_c2p = g_nonsaturating_loss(fake_feat_photo)

        gan_feat_loss = (gan_feat_loss_p2c + gan_feat_loss_c2p) / 2

        cyc_id_loss = F.mse_loss(id_net(ret_p2c_recon['result']),
                                 id_net(ret_p2c_recon['org']).detach())

        g_loss = 10 * gan_loss + 10 * cyc_loss + gan_feat_loss + gan_feat_loss + 10 * attr_loss + 10000 * cyc_id_loss

        loss_dict["gan"] = gan_loss
        loss_dict["cyc"] = cyc_loss
        loss_dict["attr"] = attr_loss
        loss_dict["feat"] = gan_feat_loss
        loss_dict["idt"] = cyc_id_loss

        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        accumulate(g_ema, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_p_val = loss_reduced["d_p"].mean().item()
        d_loss_c_val = loss_reduced["d_c"].mean().item()
        gan_loss_val = loss_reduced["gan"].mean().item()
        cyc_loss_val = loss_reduced["cyc"].mean().item()
        feat_loss_val = loss_reduced["feat"].mean().item()
        attr_loss_val = loss_reduced["attr"].mean().item()
        idt_loss_val = loss_reduced["idt"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        real_score_p_val = loss_reduced["real_score_p"].mean().item()
        fake_score_p_val = loss_reduced["fake_score_p"].mean().item()
        real_score_c_val = loss_reduced["real_score_c"].mean().item()
        fake_score_c_val = loss_reduced["fake_score_c"].mean().item()

        if get_rank() == 0:
            pbar.set_description((
                f"d_p: {d_loss_p_val:.4f}; d_c: {d_loss_c_val:.4f}; g: {gan_loss_val:.4f}, {cyc_loss_val:.4f}, {feat_loss_val:.4f}, {attr_loss_val:.4f}, {idt_loss_val:.4f}; r1: {r1_val:.4f}; "
                ))

            if i % 100 == 0:
                with torch.no_grad():
                    g_ema.eval()
                    ret = g_ema([sample_z],
                                truncation_latent=mean_latent,
                                mode='p2c')
                    utils.save_image(
                        ret['result'],
                        f"sample/{args.name}/p2c_exg_{str(i).zfill(6)}.png",
                        nrow=int(args.n_sample**0.5),
                        normalize=True,
                        range=(-1, 1),
                    )
                    utils.save_image(
                        ret['org'],
                        f"sample/{args.name}/p2c_gt_{str(i).zfill(6)}.png",
                        nrow=int(args.n_sample**0.5),
                        normalize=True,
                        range=(-1, 1),
                    )

                    ret = g_ema([sample_z],
                                truncation_latent=mean_latent,
                                mode='c2p')
                    utils.save_image(
                        ret['result'],
                        f"sample/{args.name}/c2p_exg_{str(i).zfill(6)}.png",
                        nrow=int(args.n_sample**0.5),
                        normalize=True,
                        range=(-1, 1),
                    )
                    utils.save_image(
                        ret['org'],
                        f"sample/{args.name}/c2p_gt_{str(i).zfill(6)}.png",
                        nrow=int(args.n_sample**0.5),
                        normalize=True,
                        range=(-1, 1),
                    )

            if i % 1000 == 0:
                torch.save(
                    {
                        "g": g_module.state_dict(),
                        "d_p": d_module_p.state_dict(),
                        "d_c": d_module_c.state_dict(),
                        "d_feat_p": d_module_feat_p.state_dict(),
                        "d_feat_c": d_module_feat_c.state_dict(),
                        "g_ema": g_ema.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "d_optim_p": d_optim_p.state_dict(),
                        "d_optim_c": d_optim_c.state_dict(),
                        "d_optim_fp": d_optim_fp.state_dict(),
                        "d_optim_fc": d_optim_fc.state_dict(),
                        "args": args,
                    },
                    f"checkpoint/{args.name}/{str(i).zfill(6)}.pt",
                )


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser()

    parser.add_argument('--name',
                        type=str,
                        default='temp',
                        help='name of experiment')
    parser.add_argument("--iter", type=int, default=8000000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--n_sample", type=int, default=16)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--r1", type=float, default=10)
    parser.add_argument("--path_regularize", type=float, default=2)
    parser.add_argument("--path_batch_shrink", type=int, default=2)
    parser.add_argument("--d_reg_every", type=int, default=16)
    parser.add_argument("--g_reg_every", type=int, default=4)
    parser.add_argument("--mixing", type=float, default=0.9)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--ckpt_p",
                        type=str,
                        default='./checkpoint/generator_ffhq.pt')
    parser.add_argument("--ckpt_c",
                        type=str,
                        default='./checkpoint/generator_cari.pt')
    parser.add_argument('--ckpt_pa',
                        type=str,
                        default='./checkpoint/attribute/photo_resnet.pth',
                        help='photo attr classifer checkpoint')
    parser.add_argument('--ckpt_ca',
                        type=str,
                        default='./checkpoint/attribute/cari_resnet.pth',
                        help='cari attr classifier checkpoint')
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--channel_multiplier", type=int, default=2)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--feature_loc',
                        type=int,
                        default=3,
                        help='feature location for discriminator (default: 3)')
    parser.add_argument('--freeze_D',
                        action='store_true',
                        help='freeze layers of discriminator D')

    args = parser.parse_args()

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl",
                                             init_method="env://")
        synchronize()

    args.latent = 512
    args.n_mlp = 8

    args.start_iter = 0

    generator = StyleCariGAN(
        args.size,
        args.latent,
        args.n_mlp,
        channel_multiplier=args.channel_multiplier).to(device)

    discriminator_photo = Discriminator(
        256, channel_multiplier=args.channel_multiplier).to(device)

    discriminator_cari = Discriminator(
        256, channel_multiplier=args.channel_multiplier).to(device)

    discriminator_feat_p = Discriminator_feat(
        32, channel_multiplier=args.channel_multiplier).to(device)

    discriminator_feat_c = Discriminator_feat(
        32, channel_multiplier=args.channel_multiplier).to(device)

    g_ema = StyleCariGAN(args.size,
                         args.latent,
                         args.n_mlp,
                         channel_multiplier=args.channel_multiplier).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0**g_reg_ratio, 0.99**g_reg_ratio),
    )
    d_optim_p = optim.Adam(
        discriminator_photo.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0**d_reg_ratio, 0.99**d_reg_ratio),
    )

    d_optim_c = optim.Adam(
        discriminator_cari.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0**d_reg_ratio, 0.99**d_reg_ratio),
    )

    d_optim_fp = optim.Adam(
        discriminator_feat_p.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0**d_reg_ratio, 0.99**d_reg_ratio),
    )

    d_optim_fc = optim.Adam(
        discriminator_feat_c.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0**d_reg_ratio, 0.99**d_reg_ratio),
    )

    ckpt_p = torch.load(args.ckpt_p, map_location=lambda storage, loc: storage)
    ckpt_c = torch.load(args.ckpt_c, map_location=lambda storage, loc: storage)

    generator.photo_generator.load_state_dict(ckpt_p['g_ema'], strict=True)
    generator.cari_generator.load_state_dict(ckpt_c["g_ema"], strict=False)

    discriminator_photo.load_state_dict(ckpt_p['d'])
    discriminator_cari.load_state_dict(ckpt_c['d'])

    g_ema.photo_generator.load_state_dict(ckpt_p["g_ema"], strict=False)
    g_ema.cari_generator.load_state_dict(ckpt_c["g_ema"], strict=False)

    if args.ckpt is not None:
        print("load model:", args.ckpt)

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

        generator.load_state_dict(ckpt["g"], strict=False)
        discriminator_photo.load_state_dict(ckpt["d_p"])
        discriminator_cari.load_state_dict(ckpt["d_c"])
        discriminator_feat_p.load_state_dict(ckpt["d_feat_p"])
        discriminator_feat_c.load_state_dict(ckpt["d_feat_c"])
        g_ema.load_state_dict(ckpt["g_ema"], strict=False)

        g_optim.load_state_dict(ckpt["g_optim"])
        d_optim_p.load_state_dict(ckpt["d_optim_p"])
        d_optim_c.load_state_dict(ckpt["d_optim_c"])
        d_optim_fp.load_state_dict(ckpt["d_optim_fp"])
        d_optim_fc.load_state_dict(ckpt["d_optim_fc"])
    '''
    Attribute classifier
    '''
    c_cls = ResNet18().to(device)
    p_cls = ResNet18().to(device)

    ckpt_pa = torch.load(args.ckpt_pa)
    ckpt_ca = torch.load(args.ckpt_ca)

    p_cls.load_state_dict(ckpt_pa)
    c_cls.load_state_dict(ckpt_ca)

    p_cls.eval()
    c_cls.eval()

    id_net = InceptionResnetV1(pretrained='vggface2').to(device).eval()

    with torch.no_grad():
        mean_latent = generator.photo_generator.mean_latent(4096)

    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

        discriminator_photo = nn.parallel.DistributedDataParallel(
            discriminator_photo,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

        discriminator_cari = nn.parallel.DistributedDataParallel(
            discriminator_cari,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

        discriminator_feat_p = nn.parallel.DistributedDataParallel(
            discriminator_feat_p,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

        discriminator_feat_c = nn.parallel.DistributedDataParallel(
            discriminator_feat_c,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
    ])


    train(args, generator, discriminator_photo, discriminator_cari,
          discriminator_feat_p, discriminator_feat_c, g_optim, d_optim_p,
          d_optim_c, d_optim_fp, d_optim_fc, g_ema, p_cls, c_cls, id_net,
          device)
