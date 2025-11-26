import sys
import os
import pickle
import datetime
import copy
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from lib.data_generation import generate_batches
from autoencoder_based_signal_reconstructor.autoencoder import AutoencoderBasedSignalReconstructor


class TimeStepDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X).float()  # Convert to float32 for MPS compatibility
        self.Y = torch.from_numpy(Y).float()  # Convert to float32 for MPS compatibility

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def load_and_prepare_data(data_path, sample_rate, n_alterations=10, train_val_split=0.8):
    """Load signals and prepare training/validation datasets."""
    all_batches = []
    with open(data_path, "rb") as f:
        signals = pickle.load(f)

    for signal in signals:
        batches = generate_batches(signal, sample_rate)
        all_batches.extend(batches)
        
    x = []
    y = []

    for batch in all_batches:
        original_signal = batch
        
        for _ in range(n_alterations):
            # Alter the signal
            noise_level = np.random.normal(100, 50)
            noise_level = np.clip(noise_level, 0, 200)
            phase_offset = np.random.normal(0, 0.1)
            phase_offset = np.clip(phase_offset, -0.5, 0.5)
            altered_signal = batch + np.random.normal(0, noise_level, len(batch))
            altered_signal = np.roll(altered_signal, int(phase_offset * sample_rate))

            x.append(altered_signal)
            y.append(original_signal)

    x = np.array(x)
    y = np.array(y)

    # Split into train/val
    indices = np.random.permutation(len(x))
    split_idx = int(len(x) * train_val_split)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    x_train = x[train_indices]
    y_train = y[train_indices]
    x_val = x[val_indices]
    y_val = y[val_indices]

    train_ds = TimeStepDataset(x_train, y_train)
    val_ds = TimeStepDataset(x_val, y_val)

    return train_ds, val_ds


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    """Run one epoch of training or validation."""
    model.train(train)
    total_loss = 0.0
    total_batches = 0

    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device).float()  # Convert to float32 for MPS compatibility
            logits = model(xb)
            loss = criterion(logits, yb)

            if train:
                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # clip gradients
                optimizer.step()

            total_loss += loss.item()
            total_batches += 1

    return total_loss / total_batches


def train_model(model, train_loader, val_loader, device, n_epochs=1000, 
                lr=1e-3, weight_decay=1e-4, patience=10):
    """Train the autoencoder model."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    training_record = {}

    for epoch in range(n_epochs):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        
        # Step the scheduler based on validation loss
        scheduler.step(val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:02d}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  lr={current_lr:.6f}")
        training_record[epoch] = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': current_lr
        }
        
        # Check if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            print(f"  → New best model (val_loss={val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  → No improvement for {epochs_no_improve} epoch(s)")
        
        # Early stopping
        if epochs_no_improve >= patience:
            print(f"Early stopping after {epoch+1} epochs")
            break
        
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nRestored best model with val_loss={best_val_loss:.4f}")
    
    return model, optimizer, training_record, best_val_loss


def save_model_and_plots(model, optimizer, training_record, output_dir):
    """Save the trained model and training plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = f"{output_dir}/autoencoder_based_signal_reconstructor.pth"
    torch.save({'model_state_dict': model.state_dict(), 
                'optimizer_state_dict': optimizer.state_dict()}, 
               model_path)
    print(f"Model saved to {model_path}")
    
    # Plot training curves
    epochs_list = list(training_record.keys())
    train_losses = [training_record[e]['train_loss'] for e in epochs_list]
    val_losses = [training_record[e]['val_loss'] for e in epochs_list]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_list, train_losses, label='Training', linewidth=2, color='#2E86AB')
    plt.plot(epochs_list, val_losses, label='Validation', linewidth=2, color='#A23B72')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, frameon=True, shadow=True)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plot_path = f'{output_dir}/training_validation_loss.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Training plot saved to {plot_path}")


def main():
    parser = argparse.ArgumentParser(description='Train autoencoder for signal reconstruction')
    parser.add_argument('--data-path', type=str, default='data/signals.pkl',
                        help='Path to the signals pickle file')
    parser.add_argument('--sample-rate', type=int, default=100,
                        help='Sample rate of the signals')
    parser.add_argument('--n-alterations', type=int, default=10,
                        help='Number of alterations per batch')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size for training')
    parser.add_argument('--n-epochs', type=int, default=1000,
                        help='Maximum number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience for early stopping')
    parser.add_argument('--train-val-split', type=float, default=0.8,
                        help='Train/validation split ratio')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for model and plots')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Prepare data
    print("Loading and preparing data...")
    train_ds, val_ds = load_and_prepare_data(
        args.data_path, 
        args.sample_rate, 
        args.n_alterations,
        args.train_val_split
    )
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    print(f"Training samples: {len(train_ds)}, Validation samples: {len(val_ds)}")
    
    # Setup model and device
    model = AutoencoderBasedSignalReconstructor()
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    
    # Train model
    print("\nStarting training...")
    model, optimizer, training_record, best_val_loss = train_model(
        model, train_loader, val_loader, device,
        n_epochs=args.n_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience
    )
    
    # Save results
    if args.output_dir is None:
        time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"autoencoder_based_signal_reconstructor/model_runs/{time_str}"
    else:
        output_dir = args.output_dir
    
    save_model_and_plots(model, optimizer, training_record, output_dir)
    print(f"\nTraining complete! Best validation loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
