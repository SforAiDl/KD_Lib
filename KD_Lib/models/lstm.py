import torch
import torch.nn as nn


class LSTMNet(nn.Module):
    """
    Implementation of an LSTM model for classification

    :param input_dim (int): Size of the vocabulary
    :param embed_dim (int): Embedding dimension (word vector size)
    :param hidden_dim (int): Hidden dimension for LSTM layers
    :param num_classes (int): Number of classes for classification
    :param dropout_prob (int): Dropout probability
    :param bidirectional (int): True if bidirectional LSTM needed
    :param batch_size (int): Batch size of input
    """

    def __init__(
        self,
        input_dim=100,
        embed_dim=50,
        hidden_dim=32,
        num_classes=2,
        num_layers=5,
        dropout_prob=0,
        bidirectional=False,
        pad_idx=0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embed_dim = embed_dim

        self.embedding = nn.Embedding(input_dim, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers,
            dropout=dropout_prob,
            bidirectional=bidirectional,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, x_len):
        embed_out = self.dropout(self.embedding(x))

        packed_embed_out = nn.utils.rnn.pack_padded_sequence(
            embed_out, x_len, batch_first=True, enforce_sorted=False
        )
        _, (hidden, cell) = self.lstm(packed_embed_out)

        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        fc_out = self.fc(hidden)

        return fc_out
