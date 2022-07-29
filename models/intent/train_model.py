import pandas as pd

from config.GlobalParams import MAX_SEQ_LEN
from torch.nn.utils.rnn import pad_sequence

import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch
import torch.optim as optim
from models.intent.TextCNN import TextCNN
from models.intent.TextDataset import TextDataset
from models.intent.model_utils import *
from models.intent.TextCNNHyperParams import *
import time
from utils.Preprocess import Preprocess


train_file= r"models\intent\total_train_data.csv"
data = pd.read_csv(train_file, delimiter=',')
queries = data['query'].tolist()
intents = data['intent'].tolist()

p = Preprocess(word2index_dic='train_tools\dict\chatbot_dict.bin')

sequences = []

for sentence in queries:
    pos = p.pos(sentence)
    keywords = p.get_keywords(pos, without_tag=True)
    seq = p.get_wordidx_sequence(keywords)
    seq = p.get_padding_sequence(seq, MAX_SEQ_LEN)
    sequences.append(seq)

sequences = torch.tensor(sequences)

print("Prepare Dataset")

train_ratio = 0.8
train_data_len = int(len(queries)* train_ratio)
valid_data_len = len(queries) - train_data_len

dataset = TextDataset(sequences, intents)


train_dataset, valid_dataset = random_split(dataset, [train_data_len,valid_data_len])
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)



model = TextCNN(VOCAB_SIZE, EMB_SIZE, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT_PROB, PAD_IDX)



print(f'The model has {count_parameters(model):,} trainable parameters')

optimizer = optim.Adam(model.parameters())

criterion = nn.CrossEntropyLoss()



def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    processed = 0
    model.train()
    for batch in iterator:
        data, label = batch
        optimizer.zero_grad()
        predictions = model(data).squeeze(1)
        loss = criterion(predictions, label)
        acc = accuracy(predictions, label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        processed += data.size(0)
    return epoch_loss / processed, epoch_acc / processed


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    processed = 0
    model.eval()

    with torch.no_grad():
        for batch in iterator:
            data, label = batch
            predictions = model(data).squeeze(1)
            loss = criterion(predictions, label)
            acc = accuracy(predictions, label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            processed += data.size(0)
    return epoch_loss / processed, epoch_acc / processed





N_EPOCHS = 5

best_valid_loss = float('inf')

print("Start Learning")

for epoch in range(N_EPOCHS):

    start_time = time.time()
    
    train_loss, train_acc = train(model, train_loader, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_loader, criterion)
    
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'models\intent\intent-model.pt')
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

print("Finish.")