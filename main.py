import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config
from load_data import load_codesearchnet_subset
from utils import build_vocabs
from dataset import CodeDataset
from model_rnn import Encoder, Decoder, Seq2Seq
from train import train, evaluate



CHECKPOINT_PATH = "model_rnn.pt"


def build_model(config, src_vocab, tgt_vocab):
    encoder = Encoder(
        input_dim=len(src_vocab),
        emb_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim
    )

    decoder = Decoder(
        output_dim=len(tgt_vocab),
        emb_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim
    )

    model = Seq2Seq(encoder, decoder, config)

    return model


def main():
    config = Config()

    device = torch.device("cpu")

    if os.path.exists(CHECKPOINT_PATH):
        print("Loading saved model...")

        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

        src_vocab = checkpoint["src_vocab"]
        tgt_vocab = checkpoint["tgt_vocab"]

        model = build_model(config, src_vocab, tgt_vocab)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        print("Model loaded successfully.\n")

        # Load small subset just for printing examples
        data = load_codesearchnet_subset(split="train", max_samples=5)

    else:
        print("Loading dataset...")
        full_data = load_codesearchnet_subset(split="train", max_samples=2500)

        train_data = full_data[:2000]
        val_data = full_data[2000:2500]

        print("Building vocab (from train only)...")
        src_vocab, tgt_vocab = build_vocabs(train_data, config)

        train_dataset = CodeDataset(train_data, src_vocab, tgt_vocab, config)
        val_dataset = CodeDataset(val_data, src_vocab, tgt_vocab, config)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False
        )

        print("Building model...")
        model = build_model(config, src_vocab, tgt_vocab)
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

        pad_idx = tgt_vocab.stoi[config.pad_token]
        criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

        best_val_loss = float("inf")

        train_losses = []
        val_losses = []

        print("Starting training...\n")

        for epoch in range(config.num_epochs):

            train_loss = train(
                model,
                train_loader,
                optimizer,
                criterion,
                config)
            

            val_loss = evaluate(
                model,
                val_loader,
                criterion
            )

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(
                f"Epoch {epoch+1} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss

                torch.save({
                    "model_state_dict": model.state_dict(),
                    "src_vocab": src_vocab,
                    "tgt_vocab": tgt_vocab
                }, CHECKPOINT_PATH)

                print("  -> Best model saved.")
            
        import matplotlib.pyplot as plt

        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training vs Validation Loss")
        plt.show()



    # ===== GENERATION PHASE =====

    print("=== Generation Test ===\n")

    model.eval()

    for i in range(3):
        src_text, tgt_text = data[i]

        src_tokens = src_text.strip().split()
        src_ids = src_vocab.numericalize(src_tokens)

        src_ids = src_ids[:config.max_src_len]
        src_ids += [src_vocab.stoi[config.pad_token]] * (
            config.max_src_len - len(src_ids)
        )

        src_tensor = torch.tensor(src_ids).unsqueeze(0)

        generated_ids = model.generate(
            src_tensor,
            sos_idx=tgt_vocab.stoi[config.sos_token],
            eos_idx=tgt_vocab.stoi[config.eos_token],
            max_len=config.max_tgt_len
        )

        generated_tokens = tgt_vocab.decode(generated_ids)

        print("Docstring:")
        print(src_text)
        print("\nTarget Code:")
        print(tgt_text)
        print("\nGenerated Code:")
        print(" ".join(generated_tokens))
        print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
