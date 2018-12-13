import os
import tensorflow as tf
from keras import backend as K

from config import Config as cf
from common import SquadData
from model import QANetModel

# os.environ["CUDA_VISIBLE_DEVICES"]="0"
tf_config = tf.ConfigProto(allow_soft_placement=True)
tf_config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=tf_config))

if __name__ == '__main__':
    ##### 1. Preprocessing
    # data = SquadData.load(cf.SQUAD_DATA)
    # word_vectors, char_vectors, train_ques_ids, X_train, y_train, val_ques_ids, X_valid, y_valid = SquadData.load(np_path=cf.SQUAD_NP_DATA)
    data = SquadData.load(np_path=cf.SQUAD_NP_DATA)


    ##### 2. Build model
    model = QANetModel(cf, data_train=data)
    model.summary()

    epochs = cf.EPOCH
    batch_size = cf.BATCH_SIZE
    model.train(batch_size, epochs)
