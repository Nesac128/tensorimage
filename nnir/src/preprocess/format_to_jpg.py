from PIL import Image
import os


class ConvertToJPG:
    def __init__(self,
                 image_containing_folder_path,
                 new_folder_path=None):
        # Store parameter inputs in variables
        self.icf_path = image_containing_folder_path+'/'
        self.nf_path = new_folder_path+'/'

        self.images = os.listdir(image_containing_folder_path)

    def convert(self):
        for image in self.images:
            img = Image.open(self.icf_path+image)
            rgb_img = img.convert('RGB')
            rgb_img.save(self.icf_path+image if self.nf_path is None else self.nf_path+image)
