import os
import tempfile
import argparse
from pathlib import Path
import glob
import math
import random
import shutil
import numpy as np
import torch
from torchvision import utils
import cv2
from zipfile import ZipFile
import cog
from exaggeration_model import StyleCariGAN
from align import ImageAlign
from invert import *


class Predictor(cog.Predictor):
    def setup(self):
        self.device = "cuda" if torch.cuda.is_available() else 'cpu'
        parser = argparse.ArgumentParser(description="Generate caricatures from user input images")

        parser.add_argument("--truncation", type=float, default=1, help="truncation factor")
        parser.add_argument("--truncation_mean", type=int, default=4096,
                            help="number of samples to calculate mean for truncation")
        parser.add_argument("--size", type=int, default=256, help="image sizes for generator")
        parser.add_argument('--ckpt', type=str, required=True, help='path to checkpoint')
        parser.add_argument('--input_dir', type=str, default='samples',
                            help='directory with input inverted .pt files or input images to invert')
        parser.add_argument('--output_dir', type=str, default='results', help='directory to save generated caricatures')
        parser.add_argument('--predefined_style', type=str, default="style_palette/style_palette.npy",
                            help='pre-selected style z-vector file')
        parser.add_argument('--exaggeration_factor', type=float, default=1.0, help='exaggeration factor, 0 to 1')
        parser.add_argument('--invert_images', action='store_true',
                            help='invert images in sample folder to generate caricature from them')

        # used if args.invert_images is true
        parser.add_argument("--w_iterations", type=int, default=250)
        parser.add_argument("--wp_iterations", type=int, default=2000)
        parser.add_argument("--lambda_l2", type=float, default=1)
        parser.add_argument("--lambda_p", type=float, default=1)
        parser.add_argument("--lambda_noise", type=float, default=1e5)
        parser.add_argument("--wlr", type=float, default=4e-3)
        parser.add_argument("--lr_decay_rate", type=float, default=0.2)
        parser.add_argument("--save", action='store_true')

        self.args = parser.parse_args(['--ckpt', 'checkpoint/StyleCariGAN/001000.pt', '--invert_images'])
        self.args.latent = 512
        self.args.num_layers = 14
        self.args.n_mlp = 8
        self.args.channel_multiplier = 2

        self.styles = torch.from_numpy(np.load(self.args.predefined_style)).to(self.device)  # shape N * 512
        self.styles = self.styles.unsqueeze(1)

        ckpt = torch.load(self.args.ckpt)
        self.g_ema = StyleCariGAN(
            self.args.size, self.args.latent, self.args.n_mlp, channel_multiplier=self.args.channel_multiplier
        ).to(self.device)
        self.g_ema.load_state_dict(ckpt['g_ema'], strict=False)

        if self.args.truncation < 1:
            with torch.no_grad():
                self.mean_latent = g_ema.photo_generator.mean_latent(self.args.truncation_mean)
        else:
            self.mean_latent = None

        self.align = ImageAlign()

        self.perceptual = perceptual_module().to(self.device)
        self.perceptual.eval()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)])

        n = 50000
        self.samples = 256
        w = []
        for _ in range(n // self.samples):
            sample_z = mixing_noise(self.samples, self.args.latent, 0, device=self.device)
            w.append(self.g_ema.photo_generator.style(sample_z))
        w = torch.cat(w, dim=0)
        self.args.mean_w = w.mean(dim=0)

    @cog.input("image", type=Path, help="Input image, only supports images with .png and .jpg extensions")
    @cog.input("num_samples", type=int, options=[1, 4, 9], default=1,
               help="Valid when output_type is png. Choose number of samples to view in a grid")
    @cog.input("output_type", type=str, options=['png', 'zip'], default='png',
               help="Output a png file with num_samples in a grid, or a zip file with all 64 samples")
    def predict(self, image, output_type='png', num_samples=1):
        # set input folder
        assert str(image).split('.')[-1] in ['png', 'jpg'], 'image should end with ".jpg" or ".png"'
        input_dir = 'input_cog_temp'
        os.makedirs(input_dir, exist_ok=True)
        input_path = os.path.join(input_dir, os.path.basename(str(image)))
        shutil.copy(str(image), input_path)
        self.args.input_dir = input_dir

        # --invert_images is True
        self.args.result_dir = self.args.input_dir
        self.args.image_name = os.path.basename(str(image))
        self.args.image = str(image)
        aligned_image = self.align(self.args.image)
        photo = self.transform(aligned_image).unsqueeze(0).to(self.device)
        # this is the image image without extension
        self.args.image_name = self.args.image.split('/')[-1].split('.')[0]
        invert(self.g_ema.photo_generator, self.perceptual, photo, self.device, self.args)

        inversion_file = os.path.join(self.args.input_dir, self.args.image_name + '.pt')
        self.args.current_output_dir = os.path.join(self.args.output_dir, self.args.image_name)
        generate(
            self.g_ema, self.args.truncation, self.mean_latent, inversion_file, self.styles, self.device, self.args
        )

        img_list = sorted(glob.glob(os.path.join(self.args.current_output_dir, '*')))
        if output_type == 'zip':
            out_path = Path(tempfile.mkdtemp()) / "out.zip"
            with ZipFile(str(out_path), 'w') as zip:
                for img in img_list:
                    zip.write(img)
        else:
            out_path = Path(tempfile.mkdtemp()) / "out.png"
            ran_idx = random.sample(range(0, 64), num_samples)
            # ran_idx = random.randint(0, 63)
            selected_img_list = [img_list[i] for i in ran_idx]
            image_grid = save_image_grid(int(math.sqrt(num_samples)), selected_img_list)
            cv2.imwrite(str(out_path), image_grid)
        clean_folder(input_dir)
        clean_folder(self.args.output_dir)

        return out_path


def save_image_grid(dim, images):
    assert len(images) == dim * dim, 'the number of images does not fit the grid dimensions'

    image_list = [cv2.imread(img) for img in images]
    image_grid = []
    row = 0
    for i in range(dim):
        image_grid.append(image_list[row * dim: (row + 1) * dim])
        row += 1

    final_image = cv2.vconcat([cv2.hconcat(imgs_h) for imgs_h in image_grid])
    return final_image


@torch.no_grad()
def generate(generator, truncation, truncation_latent, inversion_file, styles, device, args):
    if os.path.exists(os.path.splitext(inversion_file)[0] + '_style.pt'):
        indices = torch.load(os.path.splitext(inversion_file)[0] + '_style.pt')
    else:
        indices = range(styles.shape[0])
    inversion_file = torch.load(inversion_file)
    wp = inversion_file['wp'].to(device).unsqueeze(0)
    noise = [n.to(device) for n in inversion_file['noise']]
    os.makedirs(args.current_output_dir, exist_ok=True)

    phi = [1 - args.exaggeration_factor] * 4
    for i in indices:
        img = generator(wp, [styles[i]], noise=noise, input_is_w_plus=True, truncation=truncation,
                        truncation_latent=truncation_latent, mode='p2c', phi=phi)

        utils.save_image(
            img['result'],
            os.path.join(args.current_output_dir, f'{i}.png'),
            nrow=1,
            normalize=True,
            range=(-1, 1),
        )


def clean_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
