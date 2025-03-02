import os
import time
import torch
from tqdm import tqdm

import torchmetrics

def train_model(model, optimizer, criterion, train_loader, val_loader, num_epochs=10, unfreeze_epoch=0, device=None, model_name="default_model", scheduler=None):
    if device:
        model.to(device)
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        if epoch+1 == unfreeze_epoch:
            for param in model.parameters():
                param.requires_grad = True
            print(f"Model unfrozen at epoch {epoch+1}!")
        start_time = time.time()
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training", leave=False):
            inputs = inputs['image'].to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} Validation", leave=False):
                inputs = inputs['image'].to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        train_loss_avg = train_loss / train_total
        train_acc = train_correct / train_total
        val_loss_avg = val_loss / val_total
        val_acc = val_correct / val_total
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{num_epochs} - Time: {epoch_time:.2f}s - Train Loss: {train_loss_avg:.4f} - Train Acc: {train_acc:.4f} - Val Loss: {val_loss_avg:.4f} - Val Acc: {val_acc:.4f}")
        if (epoch + 1) % 10 == 0 or val_acc > best_val_acc:
            best_val_acc = val_acc
            save_dir = os.path.join("model_saves", model_name)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"weights_at_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"model saved as {save_path}!")
        
        if scheduler:
            scheduler.step()
