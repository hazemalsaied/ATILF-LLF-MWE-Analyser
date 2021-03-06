# ALL SETTINGS SHOULD BE WRITTEN USING MAJUSCLES
import os
import sys

CHECK_FOR_MWE = False
RIGHT_MERGE = True

XP_DEBUG_DATA_SET = False
XP_SMALL_DATA_SET = False
XP_TRAIN_DATA_SET = False
XP_CORPUS_DATA_SET = False

XP_LOAD_MODEL = False
XP_SAVE_MODEL = True
parts = os.path.dirname(os.path.abspath(__file__)).split('/')
# for i in parts[:-2]
p = '/'.join(parts[:-1])
PROJECT_PATH = p  # os.path.dirname(os.path.abspath(__file__))
VEC_TRAIN_DATA_LABELS_NUM = None
VEC_INPUT_DIMENSIONS = None

XP_SMALL_TRAIN_SENT_NUM = 2000
XP_SMALL_TEST_SENT_NUM = 500

CORPUS_USE_UNIVERSAL_POS_TAGS = True
CORPUS_SHUFFLE = True

LSTM = False
LSTM_INPUT_TIME_STEP = 4

MLP_USE_RANDOM_NORMAL_INIT = False
MLP_ADD_SPARSE_FEATURES = True
MLP_DROPOUT_1_VALUE = 0.2
MLP_USE_TANH_2 = False
MLP_USE_TANH_1 = False
MLP_WORD_EMB_LIMIT = 200
MLP_USE_VARIANCE_SCALING = False
MLP_POS_EMB_LIMIT = 25
MLP_LAYER_1_UNIT_NUM = 1024
MLP = True
MLP_USE_LEMMAS = False
MLP_STACK_ELEMS_NUM = 2
MLP_BUFFER_ELEMs_NUM = 2
MLP_LAYER_2 = True
MLP_USE_RELU_1 = True
MLP_USE_RELU_2 = True
MLP_DROPOUT_2_VALUE = 0.2
MLP_USE_LOCAL_WORD_EMBEDDING = False
MLP_LAYER_2_UNIT_NUM = 1024
MLP_DROPOUT_2 = False
MLP_DROPOUT_1 = True
MLP_USE_SIGMOID_2 = False

NN_EARLY_STOP = False
NN_VERBOSE = 2
NN_PREDICT_VERBOSE = 0
NN_BATCH_SIZE = 128
NN_EPOCHS = 15
NN_SHUFFLE = True


def toString():
    # pp = pprint.PrettyPrinter(indent=4)
    lines = []
    for key, value in globals().iteritems():
        if key == key.upper():
            lines.append(key.replace('_', ' ').lower() + ' = ' + str(value) + '\n')
    lines.sort()
    return ''.join(lines)


def load(settingDic):
    thisModule = sys.modules[__name__]
    for key, value in settingDic.iteritems():
        if key in globals():
            setattr(thisModule, key, value)
