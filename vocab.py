from collections import Counter

class Vocab:
    def __init__(self, min_freq, specials):
        self.min_freq = min_freq
        self.specials = specials
        self.itos = []
        self.stoi = {}

    def build(self, sentences):
        counter = Counter()

        for sent in sentences:
            counter.update(sent)

        self.itos = list(self.specials)

        for token, freq in counter.items():
            if freq >= self.min_freq:
                self.itos.append(token)

        self.stoi = {token: idx for idx, token in enumerate(self.itos)}

    def numericalize(self, tokens):
        return [
            self.stoi.get(token, self.stoi["<unk>"])
            for token in tokens
        ]

    def __len__(self):
        return len(self.itos)
    
    def decode(self, ids):
        return [self.itos[i] for i in ids]

