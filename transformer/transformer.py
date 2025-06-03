import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value):
        batch_size, seq_len, d_model = query.shape

        # linear projections
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        attention = F.softmax(scores, dim=-1)
        context = torch.matmul(attention, V)

        # concatenate heads
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        )

        return self.W_o(context)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # self-attention with residual connection
        attn_output = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))

        # feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class Transformer(nn.Module):
    def __init__(
        self, d_model, n_heads, n_layers, d_ff, output_size, max_len=5000, dropout=0.1
    ):
        """
        Transformer model for sequence-to-sequence tasks.

        Args:
            d_model: dimension of input vectors
            n_heads: number of attention heads
            n_layers: number of transformer blocks
            d_ff: dimension of feed-forward network
            output_size: size of output predictions
            max_len: maximum sequence length for positional encoding
            dropout: dropout rate
        """
        super().__init__()

        self.d_model = d_model
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

        self.output_projection = nn.Linear(d_model, output_size)
        self.dropout = nn.Dropout(dropout)

        self.losses = []

    def forward(self, x):
        # add positional encoding
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # output projection
        return self.output_projection(x)

    def fit(self, train_loader, criterion, optimizer, epochs=10, device="cpu"):
        """
        Train the transformer model.

        Args:
            train_loader: DataLoader with (input_sequences, target_sequences)
            criterion: loss function
            optimizer: optimizer for training
            epochs: number of training epochs
            device: device to train on ('cpu' or 'cuda')
        """
        self.train()
        self.to(device)

        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()
                output = self(data)
                loss = criterion(output.view(-1, output.size(-1)), target.view(-1))
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            self.losses.append(avg_loss)

    def predict(self, x, device="cpu"):
        """
        Make predictions with the transformer model.

        Args:
            x: input sequences tensor of shape (batch_size, seq_len, d_model)
            device: device to run inference on

        Returns:
            predictions tensor of shape (batch_size, seq_len, output_size)
        """
        self.eval()
        self.to(device)

        with torch.no_grad():
            x = x.to(device)
            return self(x)
