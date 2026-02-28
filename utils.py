def tokenize(text):
    return text.strip().split()


from vocab import Vocab

def build_vocabs(data, config):
    src_sentences = []
    tgt_sentences = []

    for src, tgt in data:
        src_sentences.append(tokenize(src))
        tgt_sentences.append(tokenize(tgt))

    specials = [
        config.pad_token,
        config.sos_token,
        config.eos_token,
        config.unk_token
    ]

    src_vocab = Vocab(config.min_freq, specials)
    tgt_vocab = Vocab(config.min_freq, specials)

    src_vocab.build(src_sentences)
    tgt_vocab.build(tgt_sentences)

    return src_vocab, tgt_vocab
