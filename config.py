class Config:
    
    max_src_len = 50
    max_tgt_len = 80
    min_freq = 2

   
    embedding_dim = 128
    hidden_dim = 256
    num_layers = 1

    
    batch_size = 16
    lr = 1e-3
    num_epochs = 10
    teacher_forcing_ratio = 0.5

 
    pad_token = "<pad>"
    sos_token = "<sos>"
    eos_token = "<eos>"
    unk_token = "<unk>"
