import click
from src.config import tensorimage_path, workspace_dir


def config():
    with open(tensorimage_path+'tensorimage/src/tmp/opt') as temp:
        opt = temp.readline()
        return opt


opt = config()

if opt == 'train':
    import src.trainer as nt

    @click.command()
    @click.argument('id_name', required=True)
    @click.argument('model_folder_name', required=True)
    @click.argument('model_name', required=True)
    @click.argument('learning_rate', required=True)
    @click.argument('n_epochs', required=True)
    @click.argument('l2_regularization_beta', required=True)
    @click.option('--train_test_split', default=0.2, help='Train-test split ratio')
    @click.option('--batch_size', default=32, help='Batch size')
    @click.option('--augment_data', default=False, help='Augment training data or not. [True/False]')
    @click.option('--cnn_architecture', default='cnn_model1')
    def train(id_name: int, model_folder_name: str, model_name: str, learning_rate: float, l2_regularization_beta: float,
              n_epochs: int, train_test_split: float, batch_size: int, augment_data: bool,
              cnn_architecture):
        trainer = nt.Train(id_name, model_folder_name, model_name, n_epochs, learning_rate, l2_regularization_beta,
                           train_test_split=train_test_split, batch_size=batch_size, augment_data=augment_data,
                           cnn_architecture=cnn_architecture)
        trainer.train_convolutional()
        trainer.write_metadata()
    train()
elif opt == 'classify':
    import src.classifier as nc

    @click.command()
    @click.argument('id_name', required=True)
    @click.argument('model_folder_name', required=True)
    @click.argument('model_name', required=True)
    @click.argument('training_dataset_name', required=True)
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
elif opt == 'add_training_dataset':
    from src.man.label_path_writer import *
    import src.image.loader as iml
    import src.image.writer as iw

    @click.command()
    @click.argument('id_name')
    @click.argument('dataset_path', required=True)
    def add_training_dataset(id_name, dataset_path):
        dataset_name = dataset_path.split('/')[-1]
        write_training_dataset_paths(dataset_path, dataset_name)
        write_labels(dataset_path, dataset_name)
        loader = iml.ImageLoader(dataset_name, 'training')
        loader.get_img_dims()
        loader.extract_image_data()
        loader.write_metadata()
        writer = iw.TrainingDataWriter(loader.image_data, 'data.csv', dataset_name, loader.img_dims, loader.MetaWriter,
                                       id_name)
        writer.write_metadata()
        writer.write_image_data()
        writer.id_man.add()
    add_training_dataset()
elif opt == 'add_unclassified_dataset':
    from src.man.label_path_writer import write_unclassified_dataset_paths
    import src.image.loader as iml
    import src.image.writer as iw

    @click.command()
    @click.argument('id_name')
    @click.argument('dataset_path', required=True)
    def add_unclassified_dataset(id_name, dataset_path):
        dataset_name = dataset_path.split('/')[-1]
        write_unclassified_dataset_paths(dataset_path, dataset_name)
        loader = iml.ImageLoader(dataset_name, 'unclassified')
        loader.get_img_dims()
        loader.extract_image_data()
        loader.write_metadata()
        data, imsize, metadata_writer = loader.image_data, loader.img_dims, loader.MetaWriter
        writer = iw.DataWriter(data, 'data.csv', dataset_name, imsize, metadata_writer, id_name)
        writer.write_metadata()
        writer.write_image_data()
    add_unclassified_dataset()
