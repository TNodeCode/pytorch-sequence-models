import torch
import torch.nn as nn
import random
from models.attention import *
from models.classifier import *

    
class CellType:
    RNN = "rnn"
    GRU = "gru"
    LSTM = "lstm"
    
    @staticmethod
    def rnn_layer(cell_type):
        cell_types = {
            "rnn": nn.RNN,
            "gru": nn.GRU,
            "lstm": nn.LSTM,
        }
        return cell_types[cell_type]
    

class RecurrentEncoder(nn.Module):
    def __init__(self, cell_type, embedding_dim, hidden_size, max_length, vocab_size, num_layers=3, dropout=0.1, bidirectional=True, use_embedding_layer=True, device='cpu'):
        super(RecurrentEncoder, self).__init__()
        self.device = device
        self.cell_type = cell_type
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_length = max_length
        self.bidirectional = bidirectional

        ### Layers ###
        if use_embedding_layer:
            self.embedding = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=embedding_dim      
            )
        else:
            self.embedding = None
        self.rnn = CellType.rnn_layer(cell_type)(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True
        )

    def initHidden(self, batch_size, device='cpu'):
        if self.bidirectional:
            num_layers = 2 * self.num_layers
        else:
            num_layers = self.num_layers
        return torch.zeros(num_layers, batch_size, self.hidden_size).to(device)

    def forward(self, x, hidden):
        if self.embedding is not None:
            # First run the input sequences through an embedding layer
            embedded = self.embedding(x)
            # Now we need to run the embeddings through the LSTM layer
            output, hidden = self.rnn(embedded, hidden)
        else:
            output, hidden = self.rnn(x.unsqueeze(1), hidden)
        return output, hidden
    

class RecurrentAttentionDecoder(torch.nn.Module):
    def __init__(self, cell_type, embedding_dim, hidden_size, vocab_size, max_length, batch_size, num_layers=3, use_embedding_layer=True, dropout=0.1, bidirectional=False, pos_encoding=False, attention_type=AttentionType.DOT, device='cpu'):
        super(RecurrentAttentionDecoder, self).__init__()
        
        # Hyperparameters
        self.device = device
        self.cell_type = cell_type
        self.max_length = max_length
        self.batch_size = batch_size
        self.use_embedding_layer = use_embedding_layer,
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.bi_factor = 2 if bidirectional else 1
        self.pos_encoding = pos_encoding
        self.attention_type = attention_type
        
        # Layers
        if use_embedding_layer:
            self.embedding = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=embedding_dim      
            )
        else:
            self.embedding = None
        self.rnn = CellType.rnn_layer(cell_type)(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True
        )
        self.attn = AttentionType.attention_layer(attention_type)(
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            max_length=max_length,
            batch_size=batch_size,
        )
        self.cls = Classifier(
            trg_vocab_size=vocab_size,
            embedding_dim=hidden_size*(self.bi_factor*2),
            softmax_dim=1,
        )
        if self.bidirectional:
            self.fc = torch.nn.Linear(2*hidden_size, vocab_size)
        else:
            self.fc = torch.nn.Linear(hidden_size, vocab_size)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x, annotations, hidden):
        if self.embedding is not None:
            # First run the input sequences together with the positions through an embedding layer
            embedded = self.embedding(x)
        else:
            embedded = x
        # Now we need to run the embeddings through the LSTM layer
        rnn_output, hidden_new = self.rnn(embedded, hidden)
        # Compute attention and context vector
        context_vector, attention = self.attn(
            hidden_new[0] if self.cell_type == CellType.LSTM else hidden_new,
            annotations
        )
        # Finally map the outputs of the LSTM layer to a probability distribution
        output = self.cls(torch.cat([rnn_output.squeeze(), context_vector.squeeze()], dim=1))
        # Return the prediction and the hidden state of the decoder
        return output, hidden_new, attention

    def initHidden(self, batch_size, device='cpu'):
        if self.bidirectional:
            num_layers = 2 * self.num_layers
        else:
            num_layers = self.num_layers
        return torch.zeros(num_layers, batch_size, self.hidden_size).to(device)
    

class Seq2SeqAttentionRNN(nn.Module):
    def __init__(
            self,
            vocab_size_in: int,
            vocab_size_out: int,
            max_length: int,
            batch_size: int,
            num_layers: int = 3,
            bidirectional: bool = True,
            embedding_dim: int = 128,
            hidden_size: int = 128,
            dropout: float = 0.1,
            cell_type: str = CellType.LSTM,
            attention_type: str = AttentionType.GENERAL,
            teacher_forcing_prob: float = 0.5,
            device: str = "cpu",
    ):
        super(Seq2SeqAttentionRNN, self).__init__()

        # Hyperparameters
        self.vocab_size_in = vocab_size_in
        self.vocab_size_out = vocab_size_out
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.cell_type = cell_type
        self.attention_type = attention_type
        self.teacher_forcing_prob = teacher_forcing_prob
        self.device = device
        
        # Create an instance of the encoder model
        self.encoder = RecurrentEncoder(
            cell_type=cell_type,
            vocab_size=vocab_size_in,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            max_length=max_length,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            device=device
        )

        # Create an instance of the decoder model
        self.decoder = RecurrentAttentionDecoder(
            cell_type=cell_type,
            attention_type=attention_type,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            vocab_size=vocab_size_out,
            max_length=max_length,
            batch_size=batch_size, 
            num_layers=num_layers,
            dropout=0.1,
            bidirectional=bidirectional,
            device=device
        )

    def forward(self, input_seqs, target_seqs):        
        # With a certain chance present the model the true predictions
        # instead of its own predictions in the next iteration
        use_teacher_forcing = random.random() < self.teacher_forcing_prob

        # Initialize the encoder hidden state and cell state with zeros
        hn = self.encoder.initHidden(input_seqs.shape[0], device=self.device)
        cn = self.encoder.initHidden(input_seqs.shape[0], device=self.device)
        hidden = (hn, cn) if self.cell_type == CellType.LSTM else hn

        # Initialize encoder outputs tensor
        last_n_states = 2 if self.bidirectional else 1
        _hidden_size = self.hidden_size * 2 if self.bidirectional else self.hidden_size
        encoder_hidden_states = torch.zeros((self.batch_size, self.max_length, _hidden_size*self.num_layers)).to(self.device)
        encoder_outputs = torch.zeros((self.batch_size, self.max_length, _hidden_size)).to(self.device)

        ####################
        #     ENCODING     #
        ####################

        # Iterate over the sequence tokens and run every word through the encoder
        for i in range(input_seqs.shape[1]):
            # Run the i-th token of the input sequence through the encoder.
            # As a result we will get the prediction (output), the hidden state (hn).
            # The hidden state and cell state will be used as inputs in the next round
            output, hidden = self.encoder(
                x=input_seqs[:, i:i+1],
                hidden=hidden
            )
            # Save encoder outputs and states for current token
            encoder_outputs[:, i:i+1, :] = output
            encoder_hidden_states[:, i, :] = concat_hidden_states(hn)

        ####################
        #     DECODING     #
        ####################

        # Here the outputs of each iteration of the decoder are saved
        decoder_outputs = []
        decoder_attention = []

        # The first token that we be presented to the model is the first token of the target sequence
        prediction = target_seqs[:, 0]

        # Iterate over words of target sequence and run words through the decoder.
        # This will produce a prediction for the next word in the sequence
        for i in range(1, target_seqs.size(1)):
            # Run token i through decoder and get word i+1 and the new hidden state as outputs
            if use_teacher_forcing:
                output, hidden, attention = self.decoder(
                    x=target_seqs[:, i-1].unsqueeze(dim=1),
                    annotations=encoder_outputs,
                    hidden=hidden
                )
            else:
                output, hidden, attention = self.decoder(
                    x=prediction.unsqueeze(dim=1),
                    annotations=encoder_outputs,
                    hidden=hidden
                )
            decoder_outputs.append(output)
            decoder_attention.append(attention[1:])
        return decoder_outputs, torch.cat(decoder_attention, dim=2)