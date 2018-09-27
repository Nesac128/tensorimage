from PIL import Image
from progress import bar

from src.pcontrol import *
from src.man.writer import *
from src.man.reader import *
from src.meta.id import ID
from src.config import *


class ImageLoader:
    def __init__(self,
                 dataset_name):
        self.dataset_name = dataset_name

        self.rgb_vals = []
        self.imsize = []

        self.path_file = external_working_directory_path+'datasets/'+dataset_name+'/paths.txt'
        reader = TXTReader(self.path_file)
        reader.read_raw()
        reader.parse()
        self.paths = reader.parsed_data

        self.n_images = len(self.paths)

        self.pixels = []

        self.id_man = ID('dataset')
        self.id_man.read()

        self.MetaWriter = JSONWriter(self.id_man.id, dataset_metafile_path)

    def non_mean_pixels(self):
        dimensions = self.get_dims()
        loading_progress = bar.Bar("Loading images: ", max=len(self.paths))
        for pixels_n in range(len(self.paths)):
            img = Image.open(self.paths[pixels_n])
            img_rgb = img.convert('RGB')
            im_pixels = []
            for x in range(dimensions[pixels_n][0]):
                for y in range(dimensions[pixels_n][1]):
                    r, g, b = img_rgb.getpixel((x, y))
                    im_pixels.append(r)
                    im_pixels.append(g)
                    im_pixels.append(b)
            loading_progress.next()
            self.pixels.append(im_pixels)

        return self.pixels

    def get_dims(self):
        sizes = []
        for n in range(self.n_images):
            img = Image.open(self.paths[n])
            sizes.append(img.size)
            self.imsize.append(img.size[0])
            self.imsize.append(img.size[1])
        print("Finished dimension extraction...")
        return sizes

    def main(self):
        self.getRGB()
        data = self.rgb_vals

        self.MetaWriter.update(path_file=self.path_file)
        self.MetaWriter.write()

        return data, self.imsize, self.MetaWriter

    def getRGB(self):
        rgb_vals = []
        pixels = self.non_mean_pixels()
        reading_progress = bar.Bar("Reading images: ", max=len(self.paths))
        for n, im_pixels in enumerate(pixels):
            rgb_vals.append(im_pixels)
            self.rgb_vals.append(im_pixels)
            reading_progress.next()
