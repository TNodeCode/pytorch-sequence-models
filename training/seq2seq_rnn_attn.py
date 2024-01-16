import torch
from torch.utils.data import DataLoader
from models.rnn import Seq2SeqAttentionRNN, CellType


class Seq2SeqAttentionRNNPredictor():
    def __init__(
            self,            
            vocab_size_in: int,
            vocab_size_out: int,
            max_length: int,
            num_layers: int = 3,
            batch_size: int = 16,
            embedding_dim: int = 128,
            hidden_size: int = 128,
            cell_type=CellType.LSTM,
            bidirectional=True,
            device: str = "cpu",
    ):
        # Hyperparameters
        self.vocab_size_in = vocab_size_in
        self.vocab_size_out = vocab_size_out
        self.max_length = max_length
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.cell_type = cell_type
        self.bidirectional = bidirectional
        self.device = device

        # Model
        self.model = Seq2SeqAttentionRNN(
            vocab_size_in=vocab_size_in,
            vocab_size_out=vocab_size_out,
            max_length=max_length,
            batch_size=batch_size,
            num_layers=num_layers,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            cell_type=cell_type,
            device=device,
        ).to(device)

    def train_one_epoch(
            self,
            epoch: int,
            dataloader: DataLoader,
            criterion,
            optimizer,
            scheduler,
            print_every: int = 1,
            device: str = "cpu",
    ):
        history = []
        accuracies = []

        # Get a batch of training data
        for batch_index, (input_seqs, target_seqs) in enumerate(dataloader):
            input_seqs = input_seqs.to(device)
            target_seqs = target_seqs.to(torch.long).to(device)

            # Set gradients of all model parameters to zero
            optimizer.zero_grad()

            # Initialize loss
            loss = 0.0
            accuracy = 0.0
            
            # Get logits for each of the two classes
            outputs, attention = self.model(
                input_seqs,
                target_seqs,
            )

            for i, output in enumerate(outputs):
                # Get the predicted classes of the model
                topv, topi = output.topk(1)
                # Add the loss of the i-th position in the target sequence to the total loss
                loss += criterion(output.squeeze(), target_seqs[:, i])
                # Compute the accuracy for the rpedicted tokens
                accuracy += float((topi.squeeze() == target_seqs[:, i]).sum() / (target_seqs.size(0)*(target_seqs.size(1)-1)))

            # Update metrics
            history.append(loss.item())
            accuracies.append(accuracy)

            # Print learning progress to the console
            if not batch_index % print_every:
                # Get the average accuracy of the last 'print_every' batches
                _accuracy = sum(accuracies[-print_every:]) / print_every
                print(f"Epoch {epoch}: loss={loss.item() / (target_seqs.size(1))}, accuracy={_accuracy}")

            # Compute gradient
            loss.backward()
            accuracy = 0.0

            # Update weights of network
            optimizer.step()

        # Adjust the learning rate
        scheduler.step()

        return history, accuracies
    
    def train(
            self,
            dataloader: DataLoader,
            epochs: int = 5,
            batch_size: int = 16,
            lr: float=1e-3,
            print_every: int = 1
    ):
        # Create an optimizer and a learning rate scheduler for the network
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

        # Loss function
        criterion = torch.nn.NLLLoss()

        history = []
        accuracies = []

        for epoch in range(epochs):
            _history, _accuracies = self.train_one_epoch(
                epoch=epoch,
                dataloader=dataloader,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                print_every=print_every,
                device=self.device,
            )

            # Save history and accuracies of epoch
            history += _history
            accuracies += _accuracies