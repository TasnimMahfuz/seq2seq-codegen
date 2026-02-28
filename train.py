import torch
import torch.nn as nn

def train(model, dataloader, optimizer, criterion, config):
    model.train()
    epoch_loss = 0

    for src, tgt in dataloader:
        optimizer.zero_grad()

        output = model(src, tgt)

        output_dim = output.shape[-1]

        output = output[:, 1:].reshape(-1, output_dim)
        tgt = tgt[:, 1:].reshape(-1)

        loss = criterion(output, tgt)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)

def evaluate(model, dataloader, criterion):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for src, tgt in dataloader:
            output = model(src, tgt)

            output_dim = output.shape[-1]

            output = output[:, 1:].reshape(-1, output_dim)
            tgt = tgt[:, 1:].reshape(-1)

            loss = criterion(output, tgt)
            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)

