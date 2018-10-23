import click
from src.config import tensorimage_path, workspace_dir


class Config:
    def __init__(self):
        with open(tensorimage_path+'tensorimage/src/tmp/opt') as temp:
            self.opt = temp.readline()


config = Config()

if config.opt == 'train':
    import src.trainer as nt

    @click.command()
    @click.argument('id_name', required=True)
    @click.argument('model_folder_name', required=True)
    @click.argument('model_name', required=True)
    @click.argument('learning_rate', required=True)
    @click.argument('n_epochs', required=True)
    @click.argument('l2_regularization_beta', required=True)
    @click.option('--optimizer', default='GradientDescent', help='Optimizer to use for training: \
                                                                 Adam(RECOMMENDED)/GradientDescent')
    @click.option('--train_test_split', default=0.2, help='Float value to split training data into training \
                    and testing set')
    @click.option('--batch_size', default=32, help='Batch size to use for training')
    @click.option('--augment_data', default=False, help='Augment training data or not. [True/False]')
    def train(id_name: int, model_folder_name: str, model_name: str, learning_rate: float, l2_regularization_beta: float,
              n_epochs: int, optimizer: str, train_test_split: float, batch_size: int, augment_data: bool):
        trainer = nt.Train(id_name, model_folder_name, model_name, n_epochs, learning_rate, l2_regularization_beta,
                           optimizer=optimizer, train_test_split=train_test_split, batch_size=batch_size,
                           augment_data=augment_data)
        trainer.train_convolutional()
    train()
elif config.opt == 'write_training_dataset_data':
    import src.image.loader as iml
    import src.image.writer as iw

    @click.command()
    @click.argument('dataset_name', required=True)
    @click.argument('filename', required=True)
    @click.argument('id_name')
    def im_man_1(dataset_name, filename: str, id_name: str):
        loader = iml.ImageLoader(dataset_name)
        loader.get_img_dims()
        loader.extract_image_data()
        loader.write_metadata()
        data, imsize, metadata_writer = loader.image_data, loader.img_dims, loader.MetaWriter
        writer = iw.TrainingDataWriter(data,
                                       filename,
                                       dataset_name,
                                       imsize,
                                       metadata_writer,
                                       id_name)
        writer.write_metadata()
        writer.write_image_data()
        writer.id_man.add()
    im_man_1()
elif config.opt == 'write_unclassified_dataset_data':
    import src.image.loader as iml
    import src.image.writer as iw

    @click.command()
    @click.argument('dataset_name', required=True)
    @click.argument('file_name', required=True)
    @click.argument('id_name', required=True)
    def im_man_2(dataset_name: str, file_name: str, id_name):
        loader = iml.ImageLoader(dataset_name)
        loader.get_img_dims()
        loader.extract_image_data()
        loader.write_metadata()
        data, imsize, metadata_writer = loader.image_data, loader.img_dims, loader.MetaWriter
        writer = iw.DataWriter(data, file_name, dataset_name, imsize, metadata_writer, id_name)
        writer.write_metadata()
        writer.write_image_data()
    im_man_2()
elif config.opt == 'classify':
    import src.classifier as nc

    @click.command()
    @click.argument('id_name', required=True)
    @click.argument('model_folder_name', required=True)
    @click.argument('model_name', required=True)
    @click.argument('training_dataset_name', required=True)
    @click.argument('prediction_dataset_name', required=True)
    @click.option('--show_images', default=False, help='Option to display all images with labels after classification \
                                                     True/False', )
    def predict(id_name: int, model_folder_name: str, model_name: str, training_dataset_name: str, show_images: str):
        if show_images == 'True' or show_images == 'true':
            show_images = True
        elif show_images == 'False' or show_images == 'false':
            show_images = False
        predicter = nc.Predict(id_name, model_folder_name, model_name, training_dataset_name, show_image=show_images)
        predicter.predict()
        predicter.match_class_id()
        predicter.write_predictions()
    predict()
elif config.opt == 'add_training_dataset':
    from src.man.label_path_writer import *
    from shutil import copytree

    @click.command()
    @click.argument('dataset_path', required=True)
    def add_training_dataset(dataset_path):
        dataset_name = dataset_path.split('/')[-1]
        write_training_dataset_paths(dataset_name)
        write_labels(dataset_name)
        copytree(dataset_path, workspace_dir+'users/training_images/')
    add_training_dataset()
elif config.opt == 'add_unclassified_dataset':
    from src.man.label_path_writer import write_unclassified_dataset_paths
    from shutil import copytree

    @click.command()
    @click.argument('dataset_path', required=True)
    def add_unclassified_dataset(dataset_path):
        dataset_name = dataset_path.split('/')[-1]
        write_unclassified_dataset_paths(dataset_name)
        copytree(dataset_path, workspace_dir+'users/unclassified_images/')
    add_unclassified_dataset()
elif config.opt == 'resize_training_dataset':
    import src.preprocess.resize as smr

    @click.command()
    @click.argument('dataset_name', required=True)
    @click.argument('width', required=True)
    @click.argument('height', required=True)
    def resize(dataset_name, width, height):
        smr.resize_training_dataset(dataset_name, (int(width), int(height)))
    resize()
elif config.opt == 'resize_unclassified_dataset':
    import src.preprocess.resize as smr

    @click.command()
    @click.argument('dataset_name', required=True)
    @click.argument('width', required=True)
    @click.argument('height', required=True)
    def resize(dataset_name, width, height):
        smr.resize_unclassified_dataset(dataset_name, (int(width), int(height)))
    resize()
