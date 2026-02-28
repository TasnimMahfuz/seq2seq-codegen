import torch
from torch.utils.data import Dataset

class CodeDataset(Dataset):
    def __init__(self, data, src_vocab, tgt_vocab, config):
        self.data = data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.config = config

    def tokenize(self, text):
        return text.strip().split()

    def pad(self, tokens, max_len, pad_idx):
        tokens = tokens[:max_len]
        return tokens + [pad_idx] * (max_len - len(tokens))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_text, tgt_text = self.data[idx]

        src_tokens = self.tokenize(src_text)
        tgt_tokens = self.tokenize(tgt_text)

        src_ids = self.src_vocab.numericalize(src_tokens)
        tgt_ids = self.tgt_vocab.numericalize(
            ["<sos>"] + tgt_tokens + ["<eos>"]
        )

        src_ids = self.pad(
            src_ids,
            self.config.max_src_len,
            self.src_vocab.stoi["<pad>"]
        )

        tgt_ids = self.pad(
            tgt_ids,
            self.config.max_tgt_len,
            self.tgt_vocab.stoi["<pad>"]
        )

        return torch.tensor(src_ids), torch.tensor(tgt_ids)
