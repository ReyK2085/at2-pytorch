
import torch
import torch.nn as nn
import torch.optim as optim

# Example training loop
def train(model, dataloader, criterion, optimizer, epochs=5):
    for epoch in range(epochs):
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    print("Training complete")
