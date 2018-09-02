from PIL import Image

import time

from nnir.pcontrol import *


class ImageLoader:
    def __init__(self,
                 dataset_name,
                 method='mean_pixels'):
        self.dataset_name = dataset_name
        self.method = method

        self.sess = Sess()
        self.sess.add()

        self.rgb_vals = []
        self.imgs = []

        self.pfile = external_working_directory_path+'datasets/'+dataset_name+'/paths.txt'

        self.pixels = []

        self.sess.ndir()

        self.Meta = MetaData()

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

    def full_pixels(self):
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
            # print(sizes)
            # exit()
        return sizes

    def main(self):
        data = self.getRGB()

        self.Meta.write(self.sess.read(), path_file=self.pfile)
        # self.Meta.write(self.sess.read(), n_columns=str(len(data[0])))

        pman = PathManager()
        pman.cpaths()
        print(len(data[0]))
        print("Finished image loading...")
        return data

    def getRGB(self):
        rgb_vals = []
        pixels = ''
        if self.method == 'mean_pixels':
            pixels = self.mean_pixels()
        elif self.method == 'non_mean_pixels':
            pixels = self.full_pixels()
        for n, im_pixels in enumerate(pixels):
            print("Reading image ", n, " out of ", len(pixels))
            rgb_vals.append(im_pixels)
            self.rgb_vals.append(im_pixels)
        return self.rgb_vals


class ConvNetsImageLoader:
    def __init__(self,
                 im_paths):

        self.sess = Sess()
        self.sess.add()

        self.rgb_vals = []
        self.imgs = []
        self.n_images = len(im_paths)

        self.pfile = im_paths
        self.open_images()

        self.pixels = []

        self.sess.ndir()

        self.Meta = MetaData()

    def open_images(self):
        reader = Reader(self.pfile)
        dat = reader.clean_read()
        for img_path in dat:
            print("Reading image: ", img_path)
            img = Image.open(img_path)
            self.imgs.append(img)
        return True

    def raw_read_pixels(self):
        raw_pixels = []
        for img in self.imgs:
            print("Loading image: ", img)
            pixels = img.load()
            raw_pixels.append(pixels)
            print(raw_pixels)
        return raw_pixels

    def clean_pixels(self):
        for n, pixels in enumerate(self.raw_read_pixels()):
            red_pixels = []
            green_pixels = []
            blue_pixels = []
            print(pixels[1, 2])
            for x in range(self.get_dims()[n][0]):
                for y in range(self.get_dims()[n][1]):
                    for rgb in pixels[x, y]:
                        red_pixels.append(rgb[0])
                        green_pixels.append(rgb[1])
                        blue_pixels.append(rgb[2])
            self.pixels.append([])
            self.pixels[n].append(red_pixels)
            self.pixels[n].append(green_pixels)
            self.pixels[n].append(blue_pixels)

        return self.pixels

    def get_dims(self):
        sizes = []
        for img in self.imgs:
            sizes.append(img.size)
        return sizes

    def main(self):
        data = self.clean_pixels()

        self.Meta.write(self.sess.read(), path_file=self.pfile)
        # self.Meta.write(self.sess.read(), n_columns=str(len(data[0])))

        pman = PathManager()
        pman.cpaths()
        print(len(data[0]))
        print("Finished image loading...")
        return data
