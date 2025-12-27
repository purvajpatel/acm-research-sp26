import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model.phode import PhoDe
from training.dataset import SpeechDataset

def train():
    # Initialize model
    model = PhoDe()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    ctc_loss = nn.CTCLoss(blank=0)

    # Dataset (placeholder — we’ll fix this next)
    dataset = SpeechDataset()
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    for epoch in range(3):
        for specs, phonemes in loader:

            outputs = model(specs)
            outputs = outputs.transpose(0, 1)

            input_lengths = torch.full(
                (specs.size(0),),
                outputs.size(0),
                dtype=torch.long
            )

            target_lengths = torch.tensor(
                [len(p) for p in phonemes],
                dtype=torch.long
            )

            targets = torch.cat(
                [torch.tensor(p, dtype=torch.long) for p in phonemes]
            )

            loss = ctc_loss(outputs, targets, input_lengths, target_lengths)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch} completed")
    
    torch.save(model.state_dict(), "phode_model.pt")
    print("Model saved to phode_model.pt")


if __name__ == "__main__":
    train()
