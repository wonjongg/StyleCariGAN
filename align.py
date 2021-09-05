import os
import sys
import bz2
import requests

# landmarks detector
import dlib

# align function
import numpy as np
import PIL
import PIL.Image


LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'

class ImageAlign:
    def __init__(self, save_aligned_image = False):
        self.save_aligned_image = save_aligned_image
        predictor_model_path = self.get_predictor_model_path()
        self.detector = dlib.get_frontal_face_detector() # cnn_face_detection_model_v1 also can be used
        self.shape_predictor = dlib.shape_predictor(predictor_model_path)

    def __call__(self, img, result_file_path=None):
        '''
        params
        img (str) - path to image to align

        returns
        PIL.Image object - aligned as ffhq dataset
        '''
        return self.align(img, result_file_path=None)

    def get_predictor_model_path(self):
        src_path = 'shape_predictor_68_face_landmarks.dat.bz2'
        dst_path = src_path[:-4]

        if not os.path.exists(src_path):
            f = requests.get(LANDMARKS_MODEL_URL, allow_redirects=True)
            open(src_path, 'wb').write(f.content)
        
        if not os.path.exists(dst_path):
            data = bz2.BZ2File(src_path).read()
            with open(dst_path, 'wb') as fp:
                fp.write(data)
        return dst_path

    def get_landmarks(self, image):
        img = dlib.load_rgb_image(image)
        dets = self.detector(img, 1)
        for detection in dets:
            face_landmarks = [(item.x, item.y) for item in self.shape_predictor(img, detection).parts()]
            return face_landmarks
        return None

    def align_function(self, src_file, face_landmarks, output_size=256, transform_size=4096, enable_padding=True):
            # Align function from FFHQ dataset pre-processing step
            # https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py

            lm = np.array(face_landmarks)
            lm_chin          = lm[0  : 17]  # left-right
            lm_eyebrow_left  = lm[17 : 22]  # left-right
            lm_eyebrow_right = lm[22 : 27]  # left-right
            lm_nose          = lm[27 : 31]  # top-down
            lm_nostrils      = lm[31 : 36]  # top-down
            lm_eye_left      = lm[36 : 42]  # left-clockwise
            lm_eye_right     = lm[42 : 48]  # left-clockwise
            lm_mouth_outer   = lm[48 : 60]  # left-clockwise
            lm_mouth_inner   = lm[60 : 68]  # left-clockwise

            # Calculate auxiliary vectors.
            eye_left     = np.mean(lm_eye_left, axis=0)
            eye_right    = np.mean(lm_eye_right, axis=0)
            eye_avg      = (eye_left + eye_right) * 0.5
            eye_to_eye   = eye_right - eye_left
            mouth_left   = lm_mouth_outer[0]
            mouth_right  = lm_mouth_outer[6]
            mouth_avg    = (mouth_left + mouth_right) * 0.5
            eye_to_mouth = mouth_avg - eye_avg

            # Choose oriented crop rectangle.
            x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
            x /= np.hypot(*x)
            x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
            y = np.flipud(x) * [-1, 1]
            c = eye_avg + eye_to_mouth * 0.1
            quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
            qsize = np.hypot(*x) * 2

            # Load in-the-wild image.
            if not os.path.isfile(src_file):
                print('\nCannot find source image. Please run "--wilds" before "--align".')
                return
            img = PIL.Image.open(src_file).convert('RGB')
            # Shrink.
            shrink = int(np.floor(qsize / output_size * 0.5))
            if shrink > 1:
                rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
                img = img.resize(rsize, PIL.Image.ANTIALIAS)
                quad /= shrink
                qsize /= shrink

            # Crop.
            border = max(int(np.rint(qsize * 0.1)), 3)
            crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
            crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
            if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
                img = img.crop(crop)
                quad -= crop[0:2]

            # Pad.
            pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
            pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
            if enable_padding and max(pad) > border - 4:
                pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
                img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'mean')
                img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
                quad += pad[:2]

            # Transform.
            img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
            if output_size < transform_size:
                img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

            # Save aligned image.
            return img

    def align(self, img, result_file_path=None):
        if not img.endswith('.jpg') and not img.endswith('.png'):
            print('Image name should end with ".jpg" or ".png"')
            return None
        if not os.path.exists(img):
            print('Image does not exist')
            return None

        landmark = self.get_landmarks(img)
        if landmark is None: 
            print('Face is not detected')
            return None

        img = self.align_function(img, landmark)
        if self.save_aligned_image:
            img.save(result_file_path if result_file_path is not None else 'result.png', 'PNG')

        return img


if __name__ == "__main__":
    print("Testing aligner")
    RAW_IMAGE_DIR = sys.argv[1]
    aligner = ImageAlign(save_aligned_image = True)
    aligner(RAW_IMAGE_DIR)
