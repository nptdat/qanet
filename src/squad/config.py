import os

class Config:
    PREFIX = 'No1'

    ############## Paths #############
    DATA_DIR = '../../data/SQUAD_Data/v1.1'
    TRAIN_JSON = os.path.join(DATA_DIR, 'train-v1.1.json')
    DEV_JSON = os.path.join(DATA_DIR, 'dev-v1.1.json')
    SQUAD_DATA = os.path.join(DATA_DIR, 'squad_processed-v1.1.pkl')
    SQUAD_NP_DATA = os.path.join(DATA_DIR, 'numpy')  # Folder containing Numpy files
    RESULT_LOG = 'log/validation_result_{}.csv'.format(PREFIX)

    # Small data
    # DATA_DIR = '../../data/SQUAD_Data/v2.0'
    # TRAIN_JSON = os.path.join(DATA_DIR, 'train-v2.0_small.json')
    # DEV_JSON = os.path.join(DATA_DIR, 'dev-v2.0.json')
    # SQUAD_DATA = os.path.join(DATA_DIR, 'squad_processed-v2.0_small.pkl')

    MODEL_PATH = 'model/qanet_{}_ep%s.h5'.format(PREFIX)
    INFERENCE_MODEL_PATH = 'model/qanet_ep20.h5'
    TEMP_MODEL_PATH = 'model/temp_weights_{}.h5'.format(PREFIX)
    TENSORBOARD_PATH = 'log/tensorboard/'

    ############## Embedding #############
    EMBEDDING_FILE = '../../data/glove/glove.6B.300d.txt'
    W_EMB_SIZE = 300

    CHAR_DIM = 64               # Dim of the char vector. In the paper, this value is 200. But it's confused with the DIM of applying Conv1D & MaxPool
    C_CONV_KERNEL = 5           # Kernel size of Conv
    C_EMB_SIZE = C_CONV_FILTERS = 200        # Numer of filters in the Conv1D layer applied on Char Embedding. It's the output of the Conv layer, so it's also the output DIM of the char emb layer.
    PAD_CHAR = ''               # PAD char use to insert into short words to ensure that all words have the same number of WORD_LEN chars

    ############## Self-Attention layers ##############
    SELF_ATTN_ACTIVATION = 'relu'
    SELF_ATTN_DROPOUT_RATE = 0.2
    SELF_ATTN_HEADS = 8

    ############## Position-wise Feed-Forward layers ##############
    FFN_ACTIVATION = 'relu'
    FFN_HIDDEN_SIZES = [2048]


    ############## Lengths #############
    CONTEXT_LEN = 400
    WORD_LEN = 16               # Max num of chars in 1 word
    QUERY_LEN = 50
    ANSWER_LEN = 30
    D_MODEL = 128
    KERNEL_SIZE = 7
    TENSORBOARD_UPDATE_FREQ = 100

    ############## Training #############
    EPOCH = 25
    BATCH_SIZE = 8
    LEARNING_RATE = 0.001
    ADAM_B1 = 0.8
    ADAM_B2 = 0.999
    ADAM_EPS = 1e-7
    EMA_DECAY = 0.999

    REGULARIZER_L2_LAMBDA = 3e-7
    WORD_EMBED_DROPOUT = 0.1
    CHAR_EMBED_DROPOUT = 0.05
    LAYER_DROPOUT_RATE = 0.1
