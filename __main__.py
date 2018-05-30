import click

import nnir.trainer as nt
import nnir.classifier as nc
import image.loader as iml
import image.writer as iw
import sound.loader as sl
import sound.writer as sw


class Config:
    def __init__(self):
        with open('tmp/opt') as temp:
            self.opt = temp.readline()


config = Config()

if config.opt == 'train':
    @click.command()
    @click.option('--sess_id', required=True, help='Integer indicating metadata values to use')
    @click.option('--model_store_path', required=True, help='Path where to store trained model')
    @click.option('--model_name', required=True, help='Trained model name')
    @click.option('--n_perceptrons_per_layer', default=(100, 51, 51, 51), help='Tuple containing number of perceptrons per hidden layer')
    @click.option('--optimizer', default='GradientDescent', help='Optimizer type')
    @click.option('--epochs', default=150, help='Number of training epochs')
    @click.option('--learning_rate', default=0.2, help='Learning rate for optimizer')
    @click.option('--train_test_split', default=0.1)
    def train(sess_id: int, model_store_path: str, model_name: str, n_perceptrons_per_layer: tuple,
              optimizer: str, epochs: int, learning_rate: float, train_test_split: float):
        trainer = nt.Train(sess_id, model_store_path, model_name,
                           optimizer=optimizer, n_perceptrons_layer=n_perceptrons_per_layer,
                           epochs=epochs, learning_rate=learning_rate, train_test_split=train_test_split)
        trainer.train()
    train()

elif config.opt == 'im_man_1':
    @click.command()
    @click.option('--path_to_path_file', required=True)
    @click.option('--method', help='Method by which to extract pixel values')
    @click.option('--file_name', required=True, help='File name for training data CSV file')
    @click.option('--label_file_path', required=True, help='Path to text file containing labels')
    def im_man_1(path_to_path_file: str, method: str, file_name: str, label_file_path: str):
        loader = iml.ImageLoader(path_to_path_file, method=method)
        data = loader.main()
        writer = iw.TrainDataWriter(data, file_name, label_file_path)
        writer.main()
    im_man_1()

elif config.opt == 'im_man_2':
    @click.command()
    @click.option('--path_to_path_file', required=True)
    @click.option('--method', required=True, help='Method by which to extract pixel values')
    @click.option('--file_name', required=True, help='File name for training data CSV file')
    def im_man_2(path_to_path_file: str, method: str, file_name: str):
        loader = iml.ImageLoader(path_to_path_file, method=method)
        data = loader.main()
        writer = iw.DataWriter(data, file_name)
        writer.main()
    im_man_2()

elif config.opt == 'snd_man_1':
    @click.command()
    @click.option('--path_to_path_file', required=True)
    @click.option('--file_name', required=True)
    def snd_man_1(path_to_path_file: str, fname: str):
        loader = sl.Loader(path_to_path_file)
        data = loader.main()
        writer = sw.DataWriter(data, fname)
        writer.main()
    snd_man_1()

elif config.opt == 'snd_man_2':
    @click.command()
    @click.option('--path_to_path_file', required=True)
    @click.option('--path_to_label_file', required=True)
    @click.option('--file_name', required=True)
    def snd_man_2(path_to_path_file: str, path_to_label_file: str, fname: str):
        loader = sl.Loader(path_to_path_file)
        data = loader.main()
        writer = sw.TrainDataWriter(data, fname, path_to_label_file)
        writer.main()
    snd_man_2()

elif config.opt == 'classify':
    @click.command()
    @click.option('--sess_id', required=True)
    @click.option('--model_path', required=True)
    @click.option('--model_name', required=True)
    @click.option('--dataset_name', required=True)
    @click.option('prediction_fname', default='predictions')
    @click.option('--show_im', default=True)
    def predict(sess_id: int, model_path: str, model_name: str, dataset_name: str, prediction_fname: str, show_im: bool):
        predicter = nc.Predict(sess_id, model_path, model_name, dataset_name,
                               prediction_fname=prediction_fname, show_im=show_im)
        predicter.main()
