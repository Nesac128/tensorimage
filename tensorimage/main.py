import click
from src.config import nnir_path


class Config:
    def __init__(self):
        with open(nnir_path+'nnir/src/tmp/opt') as temp:
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

elif config.opt == 'im_man1':
    import src.image.loader as iml
    import src.image.writer as iw

    @click.command()
    @click.argument('dataset_name', required=True)
    @click.argument('file_name', required=True)
    @click.argument('id_name')
    def im_man_1(dataset_name, file_name: str, id_name: str):
        loader = iml.ImageLoader(dataset_name)
        loader.get_img_dims()
        loader.extract_image_data()
        loader.write_metadata()
        data, imsize, metadata_writer = loader.image_data, loader.img_dims, loader.MetaWriter
        writer = iw.TrainingDataWriter(data,
                                       file_name,
                                       dataset_name,
                                       imsize,
                                       metadata_writer,
                                       id_name)
        writer.write_metadata()
        writer.write_image_data()
        writer.id_man.add()
    im_man_1()

elif config.opt == 'im_man2':
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
    @click.option('--show_image', default=True, help='Option to display all images with labels after classification \
                                                     True/False', )
    def predict(id_name: int, model_folder_name: str, model_name: str, training_dataset_name: str,
                prediction_dataset_name, show_image: str):
        if show_image == 'True' or show_image == 'true':
            show_image = True
        elif show_image == 'False' or show_image == 'false':
            show_image = False
        predicter = nc.Predict(id_name, model_folder_name, model_name, training_dataset_name, prediction_dataset_name,
                               show_image=show_image)
        predicter.predict()
        predicter.match_class_id()
        predicter.write_predictions()
    predict()
elif config.opt == 'write_paths':
    from src.man import label_path_writer as mlpw

    @click.command()
    @click.argument('main_directory_path')
    @click.argument('dataset_name')
    def path_writer(main_directory_path, dataset_name):
        mlpw.write_paths(main_directory_path, dataset_name)
    path_writer()
elif config.opt == 'write_labels':
    from src.man import label_path_writer as mlpw

    @click.command()
    @click.argument('main_directory_path')
    @click.argument('dataset_name')
    def label_writer(main_directory_path, dataset_name):
        mlpw.write_labels(main_directory_path, dataset_name)
    label_writer()
elif config.opt == 'resize':
    import src.preprocess.resize as smr

    @click.command()
    @click.argument('base_path')
    @click.argument('new_path')
    @click.argument('width')
    @click.argument('height')
    def resize(base_path, new_path, width, height):
        smr.resize(base_path, new_path, (int(width), int(height)))
    resize()
