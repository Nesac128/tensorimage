import sys

import nnir.nnir.classifier
import nnir.nnir.trainer
import nnir.nnir.image_manager


class Manager:
    def __init__(self):
        self.trainer = nnir.nnir.trainer
        self.img_man = nnir.nnir.image_manager
        self.classifier = nnir.nnir.classifier

    def operation(self):
        op = input("What (other) operation do you wish to perform? \
                            Train model on existing data and labels [T], \
                            Create CSV file to use as training data [CT], \
                            Create CSV file to use as data for prediction [CP], \
                            Begin classification process from existing CSV file [CPP], \
                            Quit [Q]: ")

        if op == 'T':
            self.train()

        elif op == 'CT':
            self.img_man1()

        elif op == 'Q':
            quit(0)

    def train(self):
        training_data_path = input("Training_data_path for CSV file: ")
        n_columns = int(input("Number of columns in training_data CSV file (including label): "))
        n_classes = int(input("Number of label types assigned: "))
        model_storage_path = input("Storage path for model training: ")
        optimizer = input("Optimizer type: ")
        n_perceptrons_per_layer = int(input("Number of perceptrons per hidden layer: "))
        epochs = int(input("Number of epochs for training: "))
        learning_rate = float(input("Neural network learning rate: "))
        train_test_data_split = float(input("Test splitting size: "))

        train = self.trainer.Train(
            training_data_path,
            n_columns,
            n_classes,
            model_storage_path,
            optimizer=optimizer,
            n_perceptrons_layer=n_perceptrons_per_layer,
            epochs=epochs,
            learning_rate=learning_rate,
            train_test_split=train_test_data_split)

        train.train()
        self.operation()

    def img_man1(self):
        # Define variables for ImageLoader class
        image_paths = input("Path to file containing image paths to manage: ")

        # Define variables for ImageTrainDataWriter class
        fname = input("CSV file name for output storage (*.csv): ") + '.csv'
        labels = input("Path to text file containing image labels: ")

        reader = self.img_man.ImageLoader(image_paths)
        data = reader.getRGB()
        writer = self.img_man.ImageTrainDataWriter(data, fname, labels)

        writer.main()
        self.operation()

    def img_man2(self):
        # Define variables for ImageLoader class
        image_paths = input("Path to file containing image paths to manage: ")

        # Define variables for ImageDataWriter class
        fname = input("CSV file name for output storage(*.csv): ") + '.csv'

        reader = self.img_man.ImageLoader(image_paths)
        data = reader.getRGB()
        writer = self.img_man.ImageDataWriter(data, fname)
        writer.main()
        self.operation()

    def classify(self):
        # Define variables for Predict class
        model_path = input("Path where training model is stored: ")
        classifier.Predict()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        pass
    else:
        man = Manager()
        man.operation()
