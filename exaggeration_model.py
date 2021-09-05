import math
import random

import torch
from torch import nn

from model import PixelNorm, EqualLinear, ConstantInput, StyledConv, ToRGB, ExaggerationLayer


class LayerSwapGenerator(nn.Module):
    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
    ):
        super().__init__()

        self.size = size

        self.style_dim = style_dim

        layers = [PixelNorm()]

        for i in range(n_mlp):
            layers.append(
                EqualLinear(style_dim,
                            style_dim,
                            lr_mul=lr_mlp,
                            activation='fused_lrelu'))

        self.style = nn.Sequential(*layers)

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.input = ConstantInput(self.channels[4])
        self.conv1 = StyledConv(self.channels[4],
                                self.channels[4],
                                3,
                                style_dim,
                                blur_kernel=blur_kernel)
        self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[4]

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2**res, 2**res]
            self.noises.register_buffer(f'noise_{layer_idx}',
                                        torch.randn(*shape))

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2**i]

            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                ))

            self.convs.append(
                StyledConv(out_channel,
                           out_channel,
                           3,
                           style_dim,
                           blur_kernel=blur_kernel))

            self.to_rgbs.append(ToRGB(out_channel, style_dim))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2

    def make_noise(self):
        device = self.input.input.device

        noises = [torch.randn(1, 1, 2**2, 2**2, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2**i, 2**i, device=device))

        return noises

    def mean_latent(self, n_latent):
        latent_in = torch.randn(n_latent,
                                self.style_dim,
                                device=self.input.input.device)
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent

    def get_latent(self, input):
        return self.style(input)

    def forward(
            self,
            styles,
            return_latents=False,
            inject_index=None,
            truncation=1,
            truncation_latent=None,
            input_is_latent=False,
            input_is_w_plus=False,
            noise=None,
            randomize_noise=True,
            feature_out=None,
            feature_skip=None,
            feature_loc=(-1, -1),
            need_skip=(False, False),
    ):
        if not input_is_latent and not input_is_w_plus:
            styles = [self.style(s) for s in styles]

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [
                    getattr(self.noises, f'noise_{i}')
                    for i in range(self.num_layers)
                ]

        if input_is_w_plus:
            latent = styles

        else:
            if truncation < 1:
                style_t = []

                for style in styles:
                    style_t.append(truncation_latent + truncation *
                                   (style - truncation_latent))

                styles = style_t

            if len(styles) < 2:
                inject_index = self.n_latent

                if styles[0].ndim < 3:
                    latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)

                else:
                    latent = styles[0]

            elif len(styles) == self.n_latent:
                styles = [style.unsqueeze(1) for style in styles]
                latent = torch.cat(styles, dim=1)

            else:
                if inject_index is None:
                    inject_index = random.randint(1, self.n_latent - 1)

                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
                latent2 = styles[1].unsqueeze(1).repeat(
                    1, self.n_latent - inject_index, 1)

                latent = torch.cat([latent, latent2], 1)

        if feature_out is None:
            out = self.input(latent)
            out = self.conv1(out, latent[:, 0], noise=noise[0])

        if not need_skip[1] and feature_loc[1] == 0:
            return out, None

        if not need_skip[0] and feature_loc[0] == 0:
            out = feature_out
            skip = feature_skip

        if feature_skip is None:
            skip = self.to_rgb1(out, latent[:, 1])

        if need_skip[1] and feature_loc[1] == 0:
            return out, skip

        if need_skip[0] and feature_loc[0] == 0:
            out = feature_out
            skip = feature_skip

        i = 1
        for j, (conv1, conv2, noise1, noise2, to_rgb) in enumerate(
                zip(self.convs[::2], self.convs[1::2], noise[1::2],
                    noise[2::2], self.to_rgbs)):
            if j + 1 < feature_loc[0]:
                i += 2
                continue

            if not j + 1 == feature_loc[0]:
                out = conv1(out, latent[:, i], noise=noise1)
                out = conv2(out, latent[:, i + 1], noise=noise2)

            if not need_skip[1] and feature_loc[1] == j + 1:
                return out, skip

            if not need_skip[0] and feature_loc[0] == j + 1:
                out = feature_out
                skip = feature_skip

            if not j + 1 == feature_loc[0] or not need_skip[0]:
                skip = to_rgb(out, latent[:, i + 2], skip)

            if need_skip[1] and feature_loc[1] == j + 1:
                return out, skip

            if need_skip[0] and feature_loc[0] == j + 1:
                out = feature_out
                skip = feature_skip

            i += 2

        image = skip

        if return_latents:
            return image, latent

        return image, None


class StyleCariGAN(nn.Module):
    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
        n_layer=4,
    ):
        super().__init__()

        self.cari_generator = LayerSwapGenerator(
            size,
            style_dim,
            n_mlp,
            channel_multiplier,
            blur_kernel,
            lr_mlp,
        )

        self.photo_generator = LayerSwapGenerator(
            size,
            style_dim,
            n_mlp,
            channel_multiplier,
            blur_kernel,
            lr_mlp,
        )

        self.deformation_blocks_PC = ExaggerationLayer(n=n_layer)
        self.deformation_blocks_CP = ExaggerationLayer(n=n_layer)

    def forward(self,
                styles,
                styles_fine=None,
                return_latents=False,
                inject_index=None,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                input_is_w_plus=False,
                noise=None,
                randomize_noise=True,
                phi=[0, 0, 0, 0],
                mode='p2c'):

        if styles_fine is None:
            styles_fine = styles

        ret = {}
        ret['po'] = []
        ret['co'] = []
        ret['ro'] = []

        if mode == 'p2c':
            po_lv1, ps_lv1 = self.photo_generator(
                styles,
                return_latents,
                inject_index=inject_index,
                truncation=truncation,
                truncation_latent=truncation_latent,
                input_is_latent=input_is_latent,
                input_is_w_plus=input_is_w_plus,
                noise=noise,
                randomize_noise=randomize_noise,
                feature_loc=(-1, 0),
            )

            co_lv1 = self.deformation_blocks_PC(po_lv1, 0)
            co_lv1 = torch.lerp(co_lv1, po_lv1, phi[0])
            ro_lv1 = self.deformation_blocks_CP(co_lv1, 0)

            ret['po'].append(po_lv1)
            ret['ro'].append(ro_lv1)

            po_lv2, ps_lv2 = self.photo_generator(
                styles,
                return_latents,
                inject_index=inject_index,
                truncation=truncation,
                truncation_latent=truncation_latent,
                input_is_latent=input_is_latent,
                input_is_w_plus=input_is_w_plus,
                noise=noise,
                randomize_noise=randomize_noise,
                feature_out=co_lv1,
                feature_skip=ps_lv1,
                feature_loc=(0, 1),
            )

            co_lv2 = self.deformation_blocks_PC(po_lv2, 1)
            co_lv2 = torch.lerp(co_lv2, po_lv2, phi[1])
            ro_lv2 = self.deformation_blocks_CP(co_lv2, 1)

            ret['po'].append(po_lv2)
            ret['ro'].append(ro_lv2)

            po_lv3, ps_lv3 = self.photo_generator(
                styles,
                return_latents,
                inject_index=inject_index,
                truncation=truncation,
                truncation_latent=truncation_latent,
                input_is_latent=input_is_latent,
                input_is_w_plus=input_is_w_plus,
                noise=noise,
                randomize_noise=randomize_noise,
                feature_out=co_lv2,
                feature_skip=ps_lv2,
                feature_loc=(1, 2),
            )

            co_lv3 = self.deformation_blocks_PC(po_lv3, 2)
            co_lv3 = torch.lerp(co_lv3, po_lv3, phi[2])
            ro_lv3 = self.deformation_blocks_CP(co_lv3, 2)

            ret['po'].append(po_lv3)
            ret['ro'].append(ro_lv3)

            po_lv4, ps_lv4 = self.photo_generator(
                styles,
                return_latents,
                inject_index=inject_index,
                truncation=truncation,
                truncation_latent=truncation_latent,
                input_is_latent=input_is_latent,
                input_is_w_plus=input_is_w_plus,
                noise=noise,
                randomize_noise=randomize_noise,
                feature_out=co_lv3,
                feature_skip=ps_lv3,
                feature_loc=(2, 3),
                need_skip=(False, True),
            )

            co_lv4 = self.deformation_blocks_PC(po_lv4, 3)
            co_lv4 = torch.lerp(co_lv4, po_lv4, phi[3])
            ro_lv4 = self.deformation_blocks_CP(co_lv4, 3)

            gt_co_lv4, gt_cs_lv4 = self.cari_generator(
                styles,
                return_latents,
                inject_index=inject_index,
                truncation=1.0,
                truncation_latent=truncation_latent,
                input_is_latent=input_is_latent,
                noise=noise,
                randomize_noise=randomize_noise,
                feature_loc=(-1, 3),
                need_skip=(True, True),
            )

            ret['co'] = co_lv4
            ret['gt_co'] = gt_co_lv4
            ret['po'].append(po_lv4)
            ret['ro'].append(ro_lv4)

            img_cari, _ = self.cari_generator(
                styles_fine,
                return_latents,
                inject_index=inject_index,
                truncation=1.0,
                truncation_latent=truncation_latent,
                input_is_latent=input_is_latent,
                noise=noise,
                randomize_noise=randomize_noise,
                feature_out=co_lv4,
                feature_skip=ps_lv4,
                feature_loc=(3, -1),
                need_skip=(True, True),
            )

            img_org, _ = self.photo_generator(
                styles,
                return_latents,
                inject_index=inject_index,
                truncation=truncation,
                truncation_latent=truncation_latent,
                input_is_latent=input_is_latent,
                input_is_w_plus=input_is_w_plus,
                noise=noise,
                randomize_noise=randomize_noise,
            )
            ret['result'] = img_cari
            ret['org'] = img_org

        elif mode == 'p2c_recon':
            po_lv1, ps_lv1 = self.photo_generator(
                styles,
                return_latents,
                inject_index=inject_index,
                truncation=truncation,
                truncation_latent=truncation_latent,
                input_is_latent=input_is_latent,
                input_is_w_plus=input_is_w_plus,
                noise=noise,
                randomize_noise=randomize_noise,
                feature_loc=(-1, 0),
            )

            co_lv1 = self.deformation_blocks_PC(po_lv1, 0)
            co_lv1 = torch.lerp(co_lv1, po_lv1, phi[0])
            ro_lv1 = self.deformation_blocks_CP(co_lv1, 0)

            ret['po'].append(po_lv1)
            ret['ro'].append(ro_lv1)

            po_lv2, ps_lv2 = self.photo_generator(
                styles,
                return_latents,
                inject_index=inject_index,
                truncation=truncation,
                truncation_latent=truncation_latent,
                input_is_latent=input_is_latent,
                input_is_w_plus=input_is_w_plus,
                noise=noise,
                randomize_noise=randomize_noise,
                feature_out=ro_lv1,
                feature_skip=ps_lv1,
                feature_loc=(0, 1),
            )

            co_lv2 = self.deformation_blocks_PC(po_lv2, 1)
            co_lv2 = torch.lerp(co_lv2, po_lv2, phi[1])
            ro_lv2 = self.deformation_blocks_CP(co_lv2, 1)

            ret['po'].append(po_lv2)
            ret['ro'].append(ro_lv2)

            po_lv3, ps_lv3 = self.photo_generator(
                styles,
                return_latents,
                inject_index=inject_index,
                truncation=truncation,
                truncation_latent=truncation_latent,
                input_is_latent=input_is_latent,
                input_is_w_plus=input_is_w_plus,
                noise=noise,
                randomize_noise=randomize_noise,
                feature_out=ro_lv2,
                feature_skip=ps_lv2,
                feature_loc=(1, 2),
            )

            co_lv3 = self.deformation_blocks_PC(po_lv3, 2)
            co_lv3 = torch.lerp(co_lv3, po_lv3, phi[2])
            ro_lv3 = self.deformation_blocks_CP(co_lv3, 2)

            ret['po'].append(po_lv3)
            ret['ro'].append(ro_lv3)

            po_lv4, ps_lv4 = self.photo_generator(
                styles,
                return_latents,
                inject_index=inject_index,
                truncation=truncation,
                truncation_latent=truncation_latent,
                input_is_latent=input_is_latent,
                input_is_w_plus=input_is_w_plus,
                noise=noise,
                randomize_noise=randomize_noise,
                feature_out=ro_lv3,
                feature_skip=ps_lv3,
                feature_loc=(2, 3),
                need_skip=(False, True),
            )

            co_lv4 = self.deformation_blocks_PC(po_lv4, 3)
            co_lv4 = torch.lerp(co_lv4, po_lv4, phi[3])
            ro_lv4 = self.deformation_blocks_CP(co_lv4, 3)

            gt_co_lv4, gt_cs_lv4 = self.cari_generator(
                styles,
                return_latents,
                inject_index=inject_index,
                truncation=1.0,
                truncation_latent=truncation_latent,
                input_is_latent=input_is_latent,
                noise=noise,
                randomize_noise=randomize_noise,
                feature_loc=(-1, 3),
                need_skip=(True, True),
            )

            ret['co'] = co_lv4
            ret['gt_co'] = gt_co_lv4
            ret['po'].append(po_lv4)
            ret['ro'].append(ro_lv4)

            img_cari, _ = self.photo_generator(
                styles,
                return_latents,
                inject_index=inject_index,
                truncation=1.0,
                truncation_latent=truncation_latent,
                input_is_latent=input_is_latent,
                noise=noise,
                randomize_noise=randomize_noise,
                feature_out=ro_lv4,
                feature_skip=ps_lv4,
                feature_loc=(3, -1),
                need_skip=(True, True),
            )

            img_org, _ = self.photo_generator(
                styles,
                return_latents,
                inject_index=inject_index,
                truncation=truncation,
                truncation_latent=truncation_latent,
                input_is_latent=input_is_latent,
                input_is_w_plus=input_is_w_plus,
                noise=noise,
                randomize_noise=randomize_noise,
            )
            ret['result'] = img_cari
            ret['org'] = img_org

        elif mode == 'c2p':
            co_lv1, cs_lv1 = self.cari_generator(
                styles,
                return_latents,
                inject_index=inject_index,
                truncation=truncation,
                truncation_latent=truncation_latent,
                input_is_latent=input_is_latent,
                input_is_w_plus=input_is_w_plus,
                noise=noise,
                randomize_noise=randomize_noise,
                feature_loc=(-1, 0),
            )

            po_lv1 = self.deformation_blocks_CP(co_lv1, 0)
            ro_lv1 = self.deformation_blocks_PC(po_lv1, 0)

            ret['co'].append(co_lv1)
            ret['ro'].append(ro_lv1)

            co_lv2, cs_lv2 = self.cari_generator(
                styles,
                return_latents,
                inject_index=inject_index,
                truncation=truncation,
                truncation_latent=truncation_latent,
                input_is_latent=input_is_latent,
                input_is_w_plus=input_is_w_plus,
                noise=noise,
                randomize_noise=randomize_noise,
                feature_out=po_lv1,
                feature_skip=cs_lv1,
                feature_loc=(0, 1),
            )

            po_lv2 = self.deformation_blocks_CP(co_lv2, 1)
            ro_lv2 = self.deformation_blocks_PC(po_lv2, 1)

            ret['co'].append(co_lv2)
            ret['ro'].append(ro_lv2)

            co_lv3, cs_lv3 = self.cari_generator(
                styles,
                return_latents,
                inject_index=inject_index,
                truncation=truncation,
                truncation_latent=truncation_latent,
                input_is_latent=input_is_latent,
                input_is_w_plus=input_is_w_plus,
                noise=noise,
                randomize_noise=randomize_noise,
                feature_out=po_lv2,
                feature_skip=cs_lv2,
                feature_loc=(1, 2),
            )

            po_lv3 = self.deformation_blocks_CP(co_lv3, 2)
            ro_lv3 = self.deformation_blocks_PC(po_lv3, 2)

            ret['co'].append(co_lv3)
            ret['ro'].append(ro_lv3)

            co_lv4, cs_lv4 = self.cari_generator(
                styles,
                return_latents,
                inject_index=inject_index,
                truncation=truncation,
                truncation_latent=truncation_latent,
                input_is_latent=input_is_latent,
                input_is_w_plus=input_is_w_plus,
                noise=noise,
                randomize_noise=randomize_noise,
                feature_out=po_lv3,
                feature_skip=cs_lv3,
                feature_loc=(2, 3),
                need_skip=(False, True),
            )

            po_lv4 = self.deformation_blocks_CP(co_lv4, 3)
            ro_lv4 = self.deformation_blocks_PC(po_lv4, 3)

            gt_po_lv4, gt_ps_lv4 = self.photo_generator(
                styles,
                return_latents,
                inject_index=inject_index,
                truncation=1.0,
                truncation_latent=truncation_latent,
                input_is_latent=input_is_latent,
                noise=noise,
                randomize_noise=randomize_noise,
                feature_loc=(-1, 3),
                need_skip=(True, True),
            )

            ret['po'] = po_lv4
            ret['gt_po'] = gt_po_lv4
            ret['co'].append(co_lv4)
            ret['ro'].append(ro_lv4)

            img_photo, _ = self.photo_generator(
                styles_fine,
                return_latents,
                inject_index=inject_index,
                truncation=1.0,
                truncation_latent=truncation_latent,
                input_is_latent=input_is_latent,
                noise=noise,
                randomize_noise=randomize_noise,
                feature_out=po_lv4,
                feature_skip=cs_lv4,
                feature_loc=(3, -1),
                need_skip=(True, True),
            )

            img_org, _ = self.cari_generator(
                styles,
                return_latents,
                inject_index=inject_index,
                truncation=truncation,
                truncation_latent=truncation_latent,
                input_is_latent=input_is_latent,
                input_is_w_plus=input_is_w_plus,
                noise=noise,
                randomize_noise=randomize_noise,
            )

            ret['result'] = img_photo
            ret['org'] = img_org

        elif mode == 'p_gt':
            img_org, _ = self.photo_generator(
                styles,
                return_latents,
                inject_index=inject_index,
                truncation=truncation,
                truncation_latent=truncation_latent,
                input_is_latent=input_is_latent,
                input_is_w_plus=input_is_w_plus,
                noise=noise,
                randomize_noise=randomize_noise,
            )

            return img_org

        elif mode == 'c_gt':
            img_org, _ = self.cari_generator(
                styles,
                return_latents,
                inject_index=inject_index,
                truncation=truncation,
                truncation_latent=truncation_latent,
                input_is_latent=input_is_latent,
                input_is_w_plus=input_is_w_plus,
                noise=noise,
                randomize_noise=randomize_noise,
            )

            return img_org

        elif mode == 'p_style':
            co, cs = self.cari_generator(
                styles,
                return_latents,
                inject_index=inject_index,
                truncation=truncation,
                truncation_latent=truncation_latent,
                input_is_latent=input_is_latent,
                input_is_w_plus=input_is_w_plus,
                noise=noise,
                randomize_noise=randomize_noise,
                feature_loc=(-1, 3),
                need_skip=(True, True),
            )

            img_photo, _ = self.photo_generator(
                styles_fine,
                return_latents,
                inject_index=inject_index,
                truncation=truncation,
                truncation_latent=truncation_latent,
                input_is_latent=input_is_latent,
                input_is_w_plus=input_is_w_plus,
                noise=noise,
                randomize_noise=randomize_noise,
                feature_out=co,
                feature_skip=cs,
                feature_loc=(3, -1),
                need_skip=(True, True),
            )

            return img_photo

        elif mode == 'c_style':
            po, ps = self.photo_generator(
                styles,
                return_latents,
                inject_index=inject_index,
                truncation=truncation,
                truncation_latent=truncation_latent,
                input_is_latent=input_is_latent,
                input_is_w_plus=input_is_w_plus,
                noise=noise,
                randomize_noise=randomize_noise,
                feature_loc=(-1, 3),
                need_skip=(True, True),
            )

            img_cari, _ = self.cari_generator(
                styles_fine,
                return_latents,
                inject_index=inject_index,
                truncation=truncation,
                truncation_latent=truncation_latent,
                input_is_latent=input_is_latent,
                input_is_w_plus=input_is_w_plus,
                noise=noise,
                randomize_noise=randomize_noise,
                feature_out=po,
                feature_skip=ps,
                feature_loc=(3, -1),
                need_skip=(True, True),
            )

            return img_cari

        else:
            raise Exception('Not supported mode.')

        return ret
