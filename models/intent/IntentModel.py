import torch
from config.GlobalParams import MAX_SEQ_LEN
from torch.nn.utils.rnn import pad_sequence
from models.intent.TextCNN import TextCNN
from models.intent.TextCNNHyperParams import *




class IntentModel:
    def __init__(self, model_path, preprocess):
      self.labels = {0: "인사", 1:"욕설", 2:"주문", 3:"예약", 4:"기타"}
      self.model = TextCNN(VOCAB_SIZE, EMB_SIZE, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT_PROB, PAD_IDX)
      self.model.load_state_dict(torch.load(model_path))
      self.p = preprocess
    def predict_class(self, query):
        pos = self.p.pos(sentence = query)

        keywords = self.p.get_keywords(pos, without_tag = True)
        seq = self.p.get_wordidx_sequence(keywords)
        sequences = [self.p.get_padding_sequence(seq, MAX_SEQ_LEN)]
        self.model.eval()
        predict = self.model(torch.tensor(sequences))
        predict_class = predict.argmax(1)
        return predict_class.numpy()[0]