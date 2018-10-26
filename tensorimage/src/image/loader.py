from PIL import Image
from progress import bar

from src.man.writer import *
from src.man.reader import *
from src.man.id import ID
from src.config import *


class ImageLoader:
    def __init__(self,
                 dataset_name,
                 dataset_type):
        # Store parameter
        self.dataset_name = dataset_name

        if dataset_type == "training":
            self.dataset_dir = "training_datasets"
        elif dataset_type == "unclassified":
            self.dataset_dir = "unclassified_datasets"

        self.path_file = workspace_dir+'user/'+self.dataset_dir+'/'+self.dataset_name+'/paths.txt'
        reader = TXTReader(self.path_file)
        reader.read_raw()
        reader.parse()
        self.image_paths = reader.parsed_data

        self.n_images = len(self.image_paths)
        self.img_dims = []
        self.image_data = []

        self.id_man = ID('dataset')
        self.id_man.read()

        self.MetaWriter = JSONWriter(str(int(self.id_man.id)+1), dataset_metafile_path)

    def extract_image_data(self):
        loading_progress = bar.Bar("Loading images: ", max=len(self.image_paths))
        for pixels_n in range(len(self.image_paths)):
            img = Image.open(self.image_paths[pixels_n])
            img_rgb = img.convert('RGB')
            im_pixels = []
            for x in range(self.img_dims[pixels_n][0]):
                for y in range(self.img_dims[pixels_n][1]):
                    r, g, b = img_rgb.getpixel((x, y))
                    im_pixels.append(r)
                    im_pixels.append(g)
                    im_pixels.append(b)
            loading_progress.next()
            self.image_data.append(im_pixels)

    def get_img_dims(self):
        for n in range(self.n_images):
            img = Image.open(self.image_paths[0])
            self.img_dims.append(img.size)

    def write_metadata(self):
        self.MetaWriter.update(path_file=self.path_file)
        self.MetaWriter.write()
