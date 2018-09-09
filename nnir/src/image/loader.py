from PIL import Image

import time
from src.pcontrol import *


class ImageLoader:
    def __init__(self,
                 dataset_name,
                 method='non_mean_pixels'):
        self.dataset_name = dataset_name
        self.method = method

        self.rgb_vals = []
        self.imgs = []
        self.imsize = []

        self.pfile = external_working_directory_path+'datasets/'+dataset_name+'/paths.txt'

        self.pixels = []

        self.Meta = MetaData()

        self.sess = Sess()
        self.sess.add()
        self.sess.ndir()

    def open_images(self):
        reader = Reader(self.pfile)
        dat = reader.clean_read()
        raw_pixels = []

        for img_path in dat:
            print("Reading image: ", img_path)
            img = Image.open(img_path)
            self.imgs.append(img)
            print("Loading image: ", img)
            pixels = img.load()
            raw_pixels.append(pixels)
            self.imsize.append(img.size)
            img.close()
        return raw_pixels

    def load_pixels(self):
        raw_pixels = []
        for img in self.imgs:
            print("Loading image: ", img)
            pixels = img.load()
            raw_pixels.append(pixels)
            print(raw_pixels)
        return raw_pixels

    def mean_pixels(self):
        print("Began mean_pixels ...")
        time.sleep(5)
        for pixels in self.load_pixels():
            im_pixels = []
            f = []
            for n_image in range(len(self.imgs)):
                f.append(n_image)
                if n_image > f[0]:
                    break
                else:
                    for x in range(self.get_dims()[n_image][0]):
                        for y in range(self.get_dims()[n_image][1]):
                            rgb_sum = pixels[x, y][0] + pixels[x, y][1] + pixels[x, y][2]
                            rgb_avr = rgb_sum / 3
                            im_pixels.append(rgb_avr)
            self.pixels.append(im_pixels)
        print("Finished mean_pixels ...")
        time.sleep(3)
        return self.pixels

    def non_mean_pixels(self):
        ims_pixels = self.open_images()
        for pixels_n in range(len(ims_pixels)):
            im_pixels = []
            print(ims_pixels[pixels_n][1, 2])
            for x in range(self.get_dims()[pixels_n][0]):
                for y in range(self.get_dims()[pixels_n][1]):
                    for ccv in ims_pixels[pixels_n][x, y]:
                        im_pixels.append(ccv)
            self.pixels.append(im_pixels)

        return self.pixels

    def get_dims(self):
        sizes = []
        for img in self.imgs:
            sizes.append(img.size)
        return sizes

    def main(self):
        data = self.getRGB()

        self.Meta.write(self.sess.read(), path_file=self.pfile)

        pman = PathManager()
        pman.cpaths()
        print(len(data[0]))
        print("Finished image loading...")
        return data, self.imsize

    def getRGB(self):
        rgb_vals = []
        pixels = ''
        if self.method == 'mean_pixels':
            pixels = self.mean_pixels()
        elif self.method == 'non_mean_pixels':
            pixels = self.non_mean_pixels()
        for n, im_pixels in enumerate(pixels):
            print("Reading image ", n, " out of ", len(pixels))
            rgb_vals.append(im_pixels)
            self.rgb_vals.append(im_pixels)
        return self.rgb_vals
