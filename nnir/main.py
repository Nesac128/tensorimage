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
    @click.argument('data_id', required=True)
    @click.argument('trained_model_path', required=True)
    @click.argument('trained_model_name', required=True)
    @click.option('--display_frequency', default=50)
    @click.option('--optimizer', default='GradientDescent')
    @click.option('--n_epochs', default=600)
    @click.option('--learning_rate', default=0.00000008)
    @click.option('--train_test_split', default=0.1)
    @click.option('--l2_regularization_beta', default=0.01)
    def train(data_id: int, trained_model_path: str, trained_model_name: str, display_frequency: int,
              optimizer: str, n_epochs: int, learning_rate: float, train_test_split: float, l2_regularization_beta: float):
        trainer = nt.Train(data_id, trained_model_path, trained_model_name,
                           optimizer=optimizer, display_frequency=display_frequency,
                           n_epochs=n_epochs, learning_rate=learning_rate, train_test_split=train_test_split,
                           l2_regularization_beta=l2_regularization_beta)
        trainer.train_convolutional()
    train()

elif config.opt == 'im_man1':
    import src.image.loader as iml
    import src.image.writer as iw

    @click.command()
    @click.argument('dataset_name', required=True)
    @click.argument('file_name', required=True)
    def im_man_1(dataset_name, file_name: str):
        loader = iml.ImageLoader(dataset_name)
        loader.get_img_dims()
        loader.extract_image_data()
        loader.write_metadata()
        data, imsize, metadata_writer = loader.image_data, loader.img_dims, loader.MetaWriter
        writer = iw.TrainingDataWriter(data, file_name, dataset_name, imsize, metadata_writer)
        writer.write_metadata()
        writer.join_data_labels()
        writer.id_man.add()
    im_man_1()

elif config.opt == 'im_man2':
    import src.image.loader as iml
    import src.image.writer as iw

    @click.command()
    @click.argument('dataset_name', required=True)
    @click.argument('method', required=True)
    @click.argument('file_name', required=True)
    def im_man_2(dataset_name: str, method: str, file_name: str):
        loader = iml.ImageLoader(dataset_name, method=method)
        data, imsize, metadata_writer = loader.main()
        writer = iw.DataWriter(data, file_name, dataset_name, imsize, metadata_writer)
        writer.main()
    im_man_2()

elif config.opt == 'classify':
    import src.classifier as nc

    @click.command()
    @click.argument('sess_id', required=True)
    @click.argument('model_path', required=True)
    @click.argument('model_name', required=True)
    @click.argument('training_dataset_name', required=True)
    @click.argument('prediction_dataset_name', required=True)
    @click.option('--prediction_fname', default='predictions')
    @click.option('--show_image', default=True)
    def predict(sess_id: int, model_path: str, model_name: str, training_dataset_name: str, prediction_dataset_name,
                prediction_fname: str, show_image: str):
        if show_image == 'True' or show_image == 'true':
            show_image = True
        elif show_image == 'False' or show_image == 'false':
            show_image = False
        predicter = nc.Predict(sess_id, model_path, model_name, training_dataset_name, prediction_dataset_name,
                               prediction_fname=prediction_fname, show_im=show_image)
        predicter.main()
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
