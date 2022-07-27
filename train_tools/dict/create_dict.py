from utils.Preprocess import Preprocess
import pickle

def read_corpus_data(filename):
    with open(filename, 'rt', encoding='UTF8') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        data = data[1:] #Remove header
    return data


corpus_data = read_corpus_data('train_tools\dict\corpus.txt')
p = Preprocess()

dict = []

for c in corpus_data:
    pos = p.pos(c[1])
    for k in pos:
        dict.append(k[0])

print(dict)

word_set = set(dict)
vocab = {word: i+2 for i, word in enumerate(word_set)}
vocab['<unk>'] = 0
vocab['<pad>'] = 1

print(vocab)

f = open("train_tools\dict\chatbot_dict.bin", "wb")

try:
    pickle.dump(vocab, f)
except Exception as e:
    print(e)
finally:
    f.close()