import torch
import torch.nn as nn
import random

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.RNN(
            emb_dim,
            hidden_dim,
            batch_first=True
        )

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, hidden = self.rnn(embedded)
        return hidden


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim):
        super().__init__()

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.RNN(
            emb_dim,
            hidden_dim,
            batch_first=True
        )
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_token, hidden):
        input_token = input_token.unsqueeze(1)

        embedded = self.embedding(input_token)
        output, hidden = self.rnn(embedded, hidden)

        prediction = self.fc_out(output.squeeze(1))

        return prediction, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, config):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.config = config

    def forward(self, src, tgt):
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        tgt_vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size)

        hidden = self.encoder(src)

        input_token = tgt[:, 0]  # <sos>

        for t in range(1, tgt_len):
            output, hidden = self.decoder(input_token, hidden)

            outputs[:, t] = output

            teacher_force = random.random() < self.config.teacher_forcing_ratio
            top1 = output.argmax(1)

            input_token = tgt[:, t] if teacher_force else top1

        return outputs
    

    def generate(self, src, sos_idx, eos_idx, max_len=80):
        self.eval()

        outputs = []

        with torch.no_grad():
            hidden = self.encoder(src)

            input_token = torch.tensor([sos_idx])

            for _ in range(max_len):
                output, hidden = self.decoder(input_token, hidden)

                top1 = output.argmax(1)

                token = top1.item()
                if token == eos_idx:
                    break

                outputs.append(token)

                input_token = top1

        return outputs

