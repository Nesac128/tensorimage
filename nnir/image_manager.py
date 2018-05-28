from PIL import Image
import time

from exceptions import *
from nnir.pcontrol import *
from meta.config import external_working_directory_path


class ImageLoader:
    def __init__(self, im_paths, method='mean_pixels'):
        self.method = method

        self.sess = Sess()
        self.sess.add()

        self.rgb_vals = []
        self.imgs = []
        self.n_images = len(im_paths)

        self.pfile = im_paths
        self.open_images()

        self.pixels = []

        if not os.path.exists('meta/sess/'+str(self.sess.read())+'/'):
            os.mkdir('meta/sess/'+str(self.sess.read())+'/')

        self.Meta = MetaData(self.sess.read())

    def open_images(self):
        reader = Reader(self.pfile)
        dat = reader.clean_read()
        for img_path in dat:
            print("Reading image: ", img_path)
            img = Image.open(img_path)
            self.imgs.append(img)
        return True

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
            for n_image in range(self.n_images):
                f.append(n_image)
                if n_image > f[0]:
                    break
                else:
                    for x, y in zip(range(self.get_dims()[n_image][0]), range(self.get_dims()[n_image][1])):
                        rgb_sum = pixels[x, y][0] + pixels[x, y][1] + pixels[x, y][2]
                        rgb_avr = rgb_sum / 3
                        im_pixels.append(rgb_avr)
            self.pixels.append(im_pixels)
        print("Finished mean_pixels ...")
        time.sleep(3)
        return self.pixels

    def cpixels(self):
        ims_pixels = self.load_pixels()
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

        self.Meta.write(path_file=self.pfile)
        self.Meta.write(n_columns=str(len(data[0])))

        pman = PathManager()
        pman.cpaths()

        return data

    def getRGB(self):
        rgb_vals = []
        pixels = ''
        if self.method == 'mean_pixels':
            pixels = self.mean_pixels()
        elif self.method == 'non_mean_pixels':
            pixels = self.cpixels()
        for n, im_pixels in enumerate(pixels):
            print("Reading image ", n, " out of ", len(pixels))
            rgb_vals.append(im_pixels)
            self.rgb_vals.append(im_pixels)
        return self.rgb_vals


class ImageDataWriter:
    def __init__(self, data, fname):
        self.raw_data = data
        self.fname = fname
        self.Meta = MetaData(Sess().read())

    def main(self):
        for image_data in self.raw_data:
            self.writeCSV(image_data)
        print(self.fname)
        self.Meta.write(data_path=os.getcwd()+'/'+self.fname)
        self.Meta.write(n_classes='0')
        self.Meta.write(trainable='False')

    def writeCSV(self, img):
        print(external_working_directory_path)
        with open(external_working_directory_path+self.fname, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(img)


class ImageTrainDataWriter:
    def __init__(self, data, fname, labels_path):
        # Store parameters in variables
        self.input_data = data
        self.fname = fname
        self.labels_path = labels_path

        self.Meta = MetaData(Sess().read())
        self.Reader = Reader(self.labels_path)

        self.labels = self.Reader.clean_read()

    def clabels(self):
        unique_labels = []
        for c, label in enumerate(self.labels):
            print(unique_labels)
            if c == 0:
                unique_labels.append(label)
            else:
                unique_labels.append(label)
                if unique_labels[c-1] == unique_labels[c]:
                    del unique_labels[c]
                else:
                    continue

        return str(len(unique_labels))

    def ccolumns(self):
        Freader = Reader(self.fname)
        data = Freader.clean_read()
        hist = []
        for invn, inv in enumerate(data):
            hist.append(inv)
            if invn == 0:
                continue
            if hist[invn] != [inv for inv in data]:
                raise ColumnNumberError("Number of columns in data file {} not matching".format(self.fname))

    def main(self, **tags):
        print("Began with TrainingDataWriting...")
        print(len(self.input_data), "Input data length")
        for imn in range(len(self.input_data)):
            print(imn, "Image-n")
            self.input_data[imn].append(self.labels[imn])
            self.writeCSV(self.input_data[imn])
        self.metaWriter(tags)
        self.ccolumns()

    def writeCSV(self, img):
        with open(external_working_directory_path+self.fname, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(img)

    def metaWriter(self, tags):
        self.Meta.write(data_path=os.getcwd() + '/' + self.fname)
        self.Meta.write(n_classes=self.clabels())
        self.Meta.write(trainable='True')
        for tag in list(tags.items()):
            exec('self.Meta.write('+str(tag[0]+'='+str(tag[1])+')'))
