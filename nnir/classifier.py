import tensorflow as tf
import pandas as pd
import numpy as np
import json

from nnir.nnir.pcontrol import *
from nnir.meta.config import external_working_directory_path
from nnir.scripts.multilayer_perceptron import multilayer_perceptron
from nnir.image.display import display_image


class Predict:
    def __init__(self,
                 sess_id: int,
                 model_path,
                 model_name,
                 dataset_name,
                 prediction_fname='predictions',
                 show_im: bool=True):
        self.id = sess_id
        self.model_path = model_path
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.prediction_fname = prediction_fname
        self.show_im = show_im

        self.raw_predictions = []
        self.predictions = []
        self.Meta = MetaData(sess_id)
        raw_meta = self.Meta.read('data_path', sess_id=sess_id)

        meta = [mt for mt in raw_meta]

        self.Reader = Reader(meta[0])

        self.pfnames = [self.prediction_fname+'.csv', self.prediction_fname+'_pfile.csv']

    def label_assigner(self):
        int_to_label = {}

        with open(external_working_directory_path+'/datasets/'+self.dataset_name+'/obj_labels.json', 'r') as ol:
            data = json.load(ol)
            ln = 0
            for item in sorted(data.values()):
                int_to_label[item] = ln
                ln += 1

        return int_to_label

    def predict(self):
        sess = tf.Session()

        # Create saver
        saver = tf.train.import_meta_graph(self.model_path + self.model_name + '.meta')

        # Attempt to restore model for prediction
        saver.restore(sess, tf.train.latest_checkpoint(self.model_path + './'))
        print("Trained model has been restored successfully!")

        x = tf.placeholder(tf.float32, [None, sess.run('n_dim:0')])

        w1, w2, w3, w4, w5 = sess.run(('weights1:0', 'weights2:0', 'weights3:0', 'weights4:0', 'weights5:0'))
        b1, b2, b3, b4, b5 = sess.run(('biases1:0', 'biases2:0', 'biases3:0', 'biases4:0', 'biases5:0'))

        weights = {
            'h1': tf.convert_to_tensor(w1),
            'h2': tf.convert_to_tensor(w2),
            'h3': tf.convert_to_tensor(w3),
            'h4': tf.convert_to_tensor(w4),
            'out': tf.convert_to_tensor(w5)
        }

        biases = {
            'b1': tf.convert_to_tensor(b1),
            'b2': tf.convert_to_tensor(b2),
            'b3': tf.convert_to_tensor(b3),
            'b4': tf.convert_to_tensor(b4),
            'out': tf.convert_to_tensor(b5)
        }

        model = multilayer_perceptron(x, weights, biases)

        prediction = sess.run(model, feed_dict={x: self.Reader.clean_read()})
        print(prediction)

        pred_labels_int = np.ndarray.tolist(sess.run(tf.argmax(prediction, axis=1)))
        self.raw_predictions = pred_labels_int

        self.int_to_label()
        print(self.predictions)
        self.write_predictions()

    def int_to_label(self):
        assigned_labels = self.label_assigner()
        print(assigned_labels)

        for raw_prediction in self.raw_predictions:
            for pred_char, pred_int in assigned_labels.items():
                if raw_prediction == pred_int:
                    self.predictions.append(pred_char)

        return self.predictions

    def write_predictions(self):
        # Re-write data in addition to predicted labels in CSV file; filename is a parameter
        data = self.Reader.clean_read()
        print(len(data))
        print(self.predictions)
        print(self.raw_predictions)
        for pix_data_n in range(len(data)):
            data[pix_data_n].append(self.predictions[pix_data_n])

        with open(self.pfnames[0], 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            for im_pix_data in data:
                writer.writerow(im_pix_data)

        # Write image paths together with predicted labels in CSV file
        raw_meta = self.Meta.read('data_path', sess_id=self.id)

        meta = [mt for mt in raw_meta]

        df = pd.read_csv(meta[0], header=None)

        raw_rows = df.iterrows()
        rows = []
        for index, row in raw_rows:
            rows.append(list(row.values))

        df = pd.read_csv('meta/sess/'+str(self.id)+'/impaths.csv', header=None)

        raw_rows = df.iterrows()
        paths = []
        for _, row in raw_rows:
            paths.append(list(row)[0])

        with open(self.pfnames[1], 'w') as pathfile:
            writer = csv.writer(pathfile, delimiter=',')
            for n in range(len(paths)):
                if self.show_im is True:
                    display_image(self.predictions[n], paths[n])
                writer.writerow([paths[n], self.predictions[n]])

    def main(self):
        self.predict()
        self.Meta.write(used_model_path__output=self.model_path +
                        self.model_name+'___' +
                        os.getcwd()+'/'+self.pfnames[0] +
                        '__'+os.getcwd()+'/'+self.pfnames[1])
