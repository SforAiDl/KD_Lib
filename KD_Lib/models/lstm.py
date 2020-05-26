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
    
	def __init__(self, input_dim=100, embed_dim=50, hidden_dim=32, num_classes=2, num_layers=5, dropout_prob=0, bidirectional=False, batch_size=32):
		super().__init__()
		self.hidden_dim = hidden_dim
		self.num_layers = num_layers
		self.batch_size = batch_size
		self.device = "cuda" if torch.cuda.is_available() else "cpu"

		self.embedding = nn.Embedding(input_dim, embed_dim, padding_idx=0)
		self.lstm = nn.LSTM(embed_dim,
							hidden_dim, 
							num_layers, 
							batch_first=True, 
							dropout=dropout_prob,
							bidirectional=bidirectional)

		self.dropout = nn.Dropout(dropout_prob)
		self.fc = nn.Linear(hidden_dim, num_classes)

		self.hidden = self.init_hidden()

	def init_hidden(self):
		weight = next(self.parameters()).data

		return (weight.new(self.num_layers, self.batch_size, self.hidden_dim).zero_().to(self.device),
				weight.new(self.num_layers, self.batch_size, self.hidden_dim).zero_().to(self.device))

	def forward(self, x):
		self.hidden = self.init_hidden()

		embed_out = self.embedding(x)
		lstm_out, self.hidden = self.lstm(embed_out, self.hidden)
		lstm_out = self.dropout(lstm_out)
		lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
		fc_out = self.fc(lstm_out)

		return fc_out

