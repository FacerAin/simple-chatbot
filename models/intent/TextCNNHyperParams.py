from utils.Preprocess import Preprocess

p = Preprocess(word2index_dic='train_tools/dict/chatbot_dict.bin')

#Hyperparameters
DROPOUT_PROB = 0.5
EMB_SIZE = 128
EPOCH = 5
N_FILTERS = 100
FILTER_SIZES = [3,4,5]
PAD_IDX = 1
OUTPUT_DIM = 5
VOCAB_SIZE = len(p.word_index) + 1
#FIXME:상수 값으로 바꿀 것. 

def TextCNNHyperParams():
    global DROPOUT_PROB, EMB_SIZE, EPOCH, N_FILTERS, FILTER_SIZES, PAD_IDX, OUTPUT_DIM, VOCAB_SIZE