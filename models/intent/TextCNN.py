import torch.nn as nn
import torch.nn.functional as F
import torch


class TextCNN(nn.Module):
    #Code Referenced by: https://happy-jihye.github.io/nlp/nlp-4/
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.conv_0 = nn.Conv2d(in_channels= 1, out_channels=n_filters, kernel_size=(filter_sizes[0], embedding_dim))
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=n_filters,kernel_size=(filter_sizes[1], embedding_dim))
        self.conv_2 = nn.Conv2d(in_channels=1, out_channels= n_filters, kernel_size=(filter_sizes[2], embedding_dim))
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.embedding(text).unsqueeze(1)
        # embedded = [batch_size, senetence_length, embedding_dim]
        # unsqueeze_embedded = [batch_size, 1, senetence_length, embedding_dim]

        conved_0 = F.relu(self.conv_0(embedded).squeeze(3))
        conved_1 = F.relu(self.conv_1(embedded).squeeze(3))
        conved_2 = F.relu(self.conv_2(embedded).squeeze(3))
        #conved = [batch_size, n_filters, sentence length - filter_size[n] + 1, 1]
        #squeeze_conved = [batch_size, n_filters, sentence length - filter_size[n] + 1]

        pooled_0 = F.max_pool1d(conved_0, conved_0.shape[2]).squeeze(2)
        pooled_1 = F.max_pool1d(conved_1, conved_1.shape[2]).squeeze(2)
        pooled_2 = F.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)
        #squeeze_pooled = [batch_size, n_filters]

        cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2),dim = 1))
        #cat = [batch_size, n_filters * len(filter_sizes)]

        return self.fc(cat)

