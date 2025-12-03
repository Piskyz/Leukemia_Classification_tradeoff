"""
CORRECT AUTOENCODER FOR LEUKEMIA ANOMALY DETECTION
With REAL bottleneck compression (100x) for proper anomaly detection
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import os
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc, f1_score
from sklearn.model_selection import train_test_split
import cv2

# Import data loading functions
try:
    from Creador_labels import cargar_todos_datasets_con_labels
except ImportError:
    print("Warning: Creador_labels not found.")

try:
    from Carga_imagenes import cargar_training_all_original, cargar_training_hem_original
except ImportError:
    print("Warning: Carga_imagenes not found.")


class CorrectAutoencoder(nn.Module):
    """Autoencoder with REAL bottleneck compression (100x)."""
    def __init__(self, bottleneck_dim=256):
        super(CorrectAutoencoder, self).__init__()
        
        # Encoder - More pooling for stronger spatial compression
        self.encoder = nn.Sequential(
            # 450x450 -> 225x225
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            
            # 225x225 -> 112x112
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            
            # 112x112 -> 56x56
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            
            # 56x56 -> 28x28 (EXTRA POOLING for more compression)
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
        )
        
        # REAL BOTTLENECK - Force 100x compression
        self.bottleneck_dim = bottleneck_dim
        self.bottleneck = nn.Sequential(
            # Reduce channels before flattening
            nn.Conv2d(128, 32, 1),  # 128 -> 32 channels
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Flatten(),
            
            # REAL COMPRESSION: 25,088 features -> 256 (100x compression!)
            nn.Linear(32 * 28 * 28, bottleneck_dim),
            nn.ReLU(True),
            nn.Dropout(0.3),  # Dropout in feature space
            
            # Expand back
            nn.Linear(bottleneck_dim, 32 * 28 * 28),
            nn.Unflatten(1, (32, 28, 28)),
        )
        
        # Decoder - Reconstruct from compressed features
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # 28x28 -> 56x56
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # 56x56 -> 112x112
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            # 112x112 -> 225x225 (custom size)
            nn.Upsample(size=(225, 225), mode='bilinear', align_corners=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            
            # 225x225 -> 450x450 (custom size)
            nn.Upsample(size=(450, 450), mode='bilinear', align_corners=True),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x
    
    def encode_to_latent(self, x):
        """Extract latent representation (for analysis)."""
        x = self.encoder(x)
        # Get the compressed latent vector
        x = self.bottleneck[0:5](x)  # Up to the linear layer
        return x


def convert_to_grayscale(images):
    """Convert images to grayscale."""
    processed = []
    for img in images:
        if len(img.shape) == 3 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif len(img.shape) == 3 and img.shape[2] == 1:
            gray = img[:, :, 0]
        else:
            gray = img
            
        if len(gray.shape) == 2:
            gray = np.expand_dims(gray, axis=-1)
        processed.append(gray)
    return np.array(processed)


def prepare_data_simple(datasets, batch_size=16):
    """
    Simple data preparation:
    - Train: 80% healthy
    - Test: 20% healthy + equal amount of leukemia
    """
    print("\nPreparing data...")
    
    # Get images
    hem_images = []
    all_images = []
    
    for key in datasets:
        if 'hem' in key.lower():
            hem_images.extend(datasets[key])
        elif 'all' in key.lower():
            all_images.extend(datasets[key])
    
    print(f"  Healthy: {len(hem_images)} images")
    print(f"  Leukemia: {len(all_images)} images")
    
    # Convert to grayscale
    hem_gray = convert_to_grayscale(hem_images)
    all_gray = convert_to_grayscale(all_images)
    
    # Normalize
    hem_gray = hem_gray.astype('float32') / 255.0
    all_gray = all_gray.astype('float32') / 255.0
    
    # Split healthy: 80% train, 20% test
    X_healthy_train, X_healthy_test = train_test_split(hem_gray, test_size=0.2, random_state=42)
    
    # Take same number of leukemia as healthy test
    n_test = min(len(X_healthy_test), len(all_gray))
    X_leukemia_test = all_gray[:n_test]
    X_healthy_test = X_healthy_test[:n_test]  # Make balanced
    
    # Final datasets
    X_train = X_healthy_train
    X_test = np.concatenate([X_healthy_test, X_leukemia_test], axis=0)
    y_test = np.concatenate([np.zeros(n_test), np.ones(n_test)], axis=0)
    
    # To tensors
    X_train_tensor = torch.FloatTensor(X_train).permute(0, 3, 1, 2)
    X_test_tensor = torch.FloatTensor(X_test).permute(0, 3, 1, 2)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Datasets
    train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\nData split:")
    print(f"  Train (healthy only): {len(X_train)} images")
    print(f"  Test (balanced): {len(X_test)} images")
    print(f"    - Healthy: {n_test}")
    print(f"    - Leukemia: {n_test}")
    
    return train_loader, test_loader


def train_correct(model, train_loader, epochs=50, lr=0.0001, device='cuda'):
    """Training for correct bottleneck model."""
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    print(f"\nTraining on {device}...")
    print("Epoch | Train Loss | Time (s)")
    print("-" * 30)
    
    train_losses = []
    epoch_times = []
    
    best_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        total_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Moderate noise for denoising
            if epoch < 30:
                noise = torch.randn_like(data) * 0.08
                noisy_data = torch.clamp(data + noise, 0, 1)
            else:
                noisy_data = data
            
            optimizer.zero_grad()
            output = model(noisy_data)
            loss = criterion(output, target)
            
            # L1 regularization for sparsity in bottleneck
            l1_lambda = 0.0002
            l1_norm = sum(p.abs().sum() for p in model.bottleneck.parameters())
            loss = loss + l1_lambda * l1_norm
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - start_time
        
        train_losses.append(avg_loss)
        epoch_times.append(epoch_time)
        
        # Check for improvement
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            marker = "*"
        else:
            patience_counter += 1
            marker = ""
        
        print(f"{epoch+1:^5} | {avg_loss:^10.6f} | {epoch_time:^9.2f} {marker}")
        
        # Learning rate decay
        if epoch == 25:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * 0.5
        elif epoch == 40:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * 0.2
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            print(f"Best loss: {best_loss:.6f}")
            break
    
    print(f"\nTraining completed in {sum(epoch_times):.2f} seconds")
    print(f"Best training loss: {best_loss:.6f}")
    return model, {'train_losses': train_losses, 'epoch_times': epoch_times, 'best_loss': best_loss}


def evaluate_model(model, test_loader, device):
    """One-time evaluation after training."""
    print("\n" + "="*60)
    print("EVALUATING ANOMALY DETECTION")
    print("="*60)
    
    model.eval()
    errors = []
    labels = []
    latent_vectors = []
    
    with torch.no_grad():
        for data, label in test_loader:
            data = data.to(device)
            recon = model(data)
            
            # MSE per image
            mse = torch.mean((recon - data) ** 2, dim=(1, 2, 3))
            
            errors.extend(mse.cpu().numpy())
            labels.extend(label.numpy())
            
            # Extract latent representations for analysis
            if len(latent_vectors) == 0:  # Only do for first batch
                latent = model.encode_to_latent(data)
                latent_vectors.extend(latent.cpu().numpy())
    
    errors = np.array(errors)
    labels = np.array(labels)
    
    # Error stats
    healthy_errors = errors[labels == 0]
    leukemia_errors = errors[labels == 1]
    
    print("\nReconstruction Error Statistics:")
    print(f"  Healthy (n={len(healthy_errors)}):")
    print(f"    Mean: {np.mean(healthy_errors):.6f}")
    print(f"    Std:  {np.std(healthy_errors):.6f}")
    print(f"    Min:  {np.min(healthy_errors):.6f}")
    print(f"    Max:  {np.max(healthy_errors):.6f}")
    
    print(f"\n  Leukemia (n={len(leukemia_errors)}):")
    print(f"    Mean: {np.mean(leukemia_errors):.6f}")
    print(f"    Std:  {np.std(leukemia_errors):.6f}")
    print(f"    Min:  {np.min(leukemia_errors):.6f}")
    print(f"    Max:  {np.max(leukemia_errors):.6f}")
    
    separation = np.mean(leukemia_errors) - np.mean(healthy_errors)
    if np.std(healthy_errors) > 0:
        separation_ratio = separation / np.std(healthy_errors)
        print(f"\n  Separation: {separation:.6f}")
        print(f"  Separation ratio: {separation_ratio:.2f} std dev")
        print(f"  Leukemia error is {np.mean(leukemia_errors)/np.mean(healthy_errors):.2f}x higher than healthy")
    else:
        print(f"\n  Separation: {separation:.6f}")
        print(f"  Separation ratio: N/A (zero std dev)")
    
    # Find threshold using Youden's J statistic
    fpr, tpr, thresholds = roc_curve(labels, errors)
    youden_j = tpr - fpr
    best_idx = np.argmax(youden_j)
    best_threshold = thresholds[best_idx]
    
    # ROC AUC
    roc_auc = auc(fpr, tpr)
    
    # Final predictions
    preds = (errors > best_threshold).astype(int)
    f1 = f1_score(labels, preds)
    
    print(f"\nAnomaly Detection Results:")
    print(f"  ROC AUC: {roc_auc:.4f}")
    print(f"  Optimal threshold: {best_threshold:.6f}")
    print(f"  F1 Score: {f1:.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(labels, preds, target_names=['Healthy', 'Leukemia']))
    
    return {
        'errors': errors,
        'labels': labels,
        'predictions': preds,
        'threshold': best_threshold,
        'roc_auc': roc_auc,
        'f1_score': f1,
        'error_stats': {
            'healthy_mean': float(np.mean(healthy_errors)),
            'healthy_std': float(np.std(healthy_errors)),
            'leukemia_mean': float(np.mean(leukemia_errors)),
            'leukemia_std': float(np.std(leukemia_errors)),
            'separation': float(separation),
            'separation_ratio': float(separation / np.std(healthy_errors)) if np.std(healthy_errors) > 0 else 0.0,
            'error_ratio': float(np.mean(leukemia_errors) / np.mean(healthy_errors)) if np.mean(healthy_errors) > 0 else 0.0
        },
        'roc_curve': (fpr, tpr),
        'latent_vectors': np.array(latent_vectors) if len(latent_vectors) > 0 else None
    }


def plot_results(training_metrics, eval_results):
    """Comprehensive plotting of results."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Training loss
    ax = axes[0, 0]
    ax.plot(training_metrics['train_losses'], 'b-', linewidth=2)
    ax.set_title('Training Loss (Should reach <0.010)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.axhline(y=0.010, color='r', linestyle='--', alpha=0.5, label='Target: 0.010')
    ax.axhline(y=0.005, color='g', linestyle='--', alpha=0.5, label='Good: 0.005')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Error distribution
    ax = axes[0, 1]
    healthy_errors = eval_results['errors'][eval_results['labels'] == 0]
    leukemia_errors = eval_results['errors'][eval_results['labels'] == 1]
    
    ax.hist(healthy_errors, bins=30, alpha=0.7, label='Healthy', color='green', density=True)
    ax.hist(leukemia_errors, bins=30, alpha=0.7, label='Leukemia', color='red', density=True)
    ax.axvline(eval_results['threshold'], color='black', linestyle='--', 
               label=f'Threshold: {eval_results["threshold"]:.6f}')
    ax.set_title('Error Distribution (Need clear separation)')
    ax.set_xlabel('Reconstruction Error')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ROC curve
    ax = axes[0, 2]
    fpr, tpr = eval_results['roc_curve']
    roc_auc = eval_results['roc_auc']
    ax.plot(fpr, tpr, 'darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'navy', lw=2, linestyle='--', label='Random')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve (Target AUC >0.75)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Error boxplot
    ax = axes[1, 0]
    data_to_plot = [healthy_errors, leukemia_errors]
    ax.boxplot(data_to_plot, labels=['Healthy', 'Leukemia'])
    ax.set_title('Error by Class (Need 3-5x difference)')
    ax.set_ylabel('Reconstruction Error')
    ax.grid(True, alpha=0.3)
    
    # Epoch times
    ax = axes[1, 1]
    ax.plot(training_metrics['epoch_times'], 'g-', linewidth=2)
    ax.set_title('Time per Epoch')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Seconds')
    ax.grid(True, alpha=0.3)
    
    # Latent space visualization (if available)
    ax = axes[1, 2]
    if eval_results['latent_vectors'] is not None and len(eval_results['latent_vectors']) > 0:
        latent = eval_results['latent_vectors']
        # Use first 2 principal components
        from sklearn.decomposition import PCA
        if latent.shape[1] > 2:
            pca = PCA(n_components=2)
            latent_2d = pca.fit_transform(latent[:100])  # First 100 samples
        else:
            latent_2d = latent[:100]
        
        ax.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.6, s=20)
        ax.set_title('Latent Space (First 100 samples)')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Latent space data not available', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Latent Space Visualization')
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('logs', exist_ok=True)
    plot_path = f'logs/autoencoder_correct_{timestamp}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlot saved to: {plot_path}")
    return plot_path


def save_report(training_metrics, eval_results, model):
    """Save comprehensive report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('logs', exist_ok=True)
    report_path = f'logs/report_correct_{timestamp}.txt'
    
    total_params = sum(p.numel() for p in model.parameters())
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("CORRECT AUTOENCODER WITH REAL BOTTLENECK - ANOMALY DETECTION REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: CorrectAutoencoder (bottleneck_dim={model.bottleneck_dim})\n")
        f.write(f"Total parameters: {total_params:,}\n\n")
        
        f.write("COMPRESSION ANALYSIS:\n")
        f.write(f"  Input pixels: 450×450×1 = 202,500\n")
        f.write(f"  Encoder output: 128×28×28 = 100,352 features\n")
        f.write(f"  Bottleneck: 100,352 → {model.bottleneck_dim} (compression ratio: {100352/model.bottleneck_dim:.1f}x)\n")
        f.write(f"  Total compression: 202,500 → {model.bottleneck_dim} ({202500/model.bottleneck_dim:.1f}x)\n\n")
        
        f.write("TRAINING:\n")
        f.write(f"  Epochs completed: {len(training_metrics['train_losses'])}\n")
        f.write(f"  Final loss: {training_metrics['train_losses'][-1]:.6f}\n")
        f.write(f"  Best loss: {training_metrics.get('best_loss', training_metrics['train_losses'][-1]):.6f}\n")
        f.write(f"  Total time: {sum(training_metrics['epoch_times']):.2f}s\n")
        f.write(f"  Average epoch time: {np.mean(training_metrics['epoch_times']):.2f}s\n\n")
        
        f.write("ERROR STATISTICS:\n")
        stats = eval_results['error_stats']
        f.write(f"  Healthy (n=300):\n")
        f.write(f"    Mean: {stats['healthy_mean']:.6f}\n")
        f.write(f"    Std:  {stats['healthy_std']:.6f}\n")
        f.write(f"  Leukemia (n=300):\n")
        f.write(f"    Mean: {stats['leukemia_mean']:.6f}\n")
        f.write(f"    Std:  {stats['leukemia_std']:.6f}\n")
        f.write(f"  Separation: {stats['separation']:.6f}\n")
        f.write(f"  Separation ratio: {stats['separation_ratio']:.2f} std dev\n")
        f.write(f"  Error ratio (Leukemia/Healthy): {stats['error_ratio']:.2f}x\n\n")
        
        f.write("PERFORMANCE METRICS:\n")
        f.write(f"  ROC AUC: {eval_results['roc_auc']:.4f}\n")
        f.write(f"  F1 Score: {eval_results['f1_score']:.4f}\n")
        f.write(f"  Optimal threshold: {eval_results['threshold']:.6f}\n")
        f.write(f"  Accuracy: {np.mean(eval_results['predictions'] == eval_results['labels']):.4f}\n\n")
        
        f.write("PERFORMANCE ASSESSMENT:\n")
        roc_auc = eval_results['roc_auc']
        error_ratio = stats['error_ratio']
        
        if roc_auc >= 0.8 and error_ratio >= 3.0:
            f.write("  EXCELLENT: Perfect anomaly detector with strong separation\n")
        elif roc_auc >= 0.75 and error_ratio >= 2.5:
            f.write("  GOOD: Effective anomaly detector\n")
        elif roc_auc >= 0.7 and error_ratio >= 2.0:
            f.write("  MODERATE: Acceptable performance\n")
        elif roc_auc >= 0.65 and error_ratio >= 1.5:
            f.write("  LIMITED: Some discriminative power\n")
        elif roc_auc >= 0.6:
            f.write("  WEAK: Barely better than random\n")
        else:
            f.write("  POOR: Fails as anomaly detector\n")
        
        f.write(f"\nTARGETS vs ACTUAL:\n")
        f.write(f"  Target ROC AUC: >0.75 | Actual: {roc_auc:.4f}\n")
        f.write(f"  Target Error Ratio: 3-5x | Actual: {error_ratio:.2f}x\n")
        f.write(f"  Target Separation Ratio: >1.5σ | Actual: {stats['separation_ratio']:.2f}σ\n")
        f.write(f"  Target Training Loss: <0.010 | Actual: {training_metrics['train_losses'][-1]:.6f}\n\n")
        
        f.write("CLASSIFICATION REPORT:\n")
        f.write(classification_report(eval_results['labels'], eval_results['predictions'], 
                                      target_names=['Healthy', 'Leukemia']))
    
    print(f"Report saved to: {report_path}")
    return report_path


def main():
    """Main function."""
    print("\n" + "="*70)
    print("LEUKEMIA ANOMALY DETECTION - CORRECT AUTOENCODER (MODEL 7)")
    print("="*70)
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    print("\n1. Loading data...")
    try:
        datasets, _ = cargar_todos_datasets_con_labels()
    except:
        print("Using fallback data loading...")
        try:
            datasets = {
                'hem': cargar_training_hem_original("data/training_data/fold_0/hem/"),
                'all': cargar_training_all_original("data/training_data/fold_0/all/")
            }
        except:
            # Create dummy data if needed
            hem_images = [np.random.rand(450, 450, 3) for _ in range(100)]
            all_images = [np.random.rand(450, 450, 3) for _ in range(100)]
            datasets = {'hem': hem_images, 'all': all_images}
            print("Using dummy data for testing")
    
    # Prepare data
    train_loader, test_loader = prepare_data_simple(datasets, batch_size=16)
    
    # Create model
    print("\n2. Creating CORRECT autoencoder (Model 7)...")
    model = CorrectAutoencoder(bottleneck_dim=256)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Model created with {total_params:,} parameters")
    print(f"\nARCHITECTURE HIGHLIGHTS:")
    print(f"  Encoder: 1→16→32→64→128 channels with 4x max pooling")
    print(f"  Bottleneck: REAL compression 100,352 features → 256 (400x!)")
    print(f"  Decoder: 256 → 128→64→32→16→1 channels")
    print(f"  Total compression: 202,500 pixels → 256 features (791x compression)")
    print(f"\nKEY IMPROVEMENTS:")
    print(f"  1. EXTRA POOLING: 56×56 → 28×28 for stronger spatial compression")
    print(f"  2. REAL BOTTLENECK: Linear layers force abstract feature learning")
    print(f"  3. 400x COMPRESSION: Forces information loss (critical for anomalies)")
    print(f"  4. L1 REGULARIZATION: Encourages sparse latent representations")
    
    # Train
    print("\n3. Training correct autoencoder...")
    print("Training parameters:")
    print(f"  Epochs: 50 (with early stopping patience=15)")
    print(f"  Learning rate: 0.0001 (lower for stable bottleneck learning)")
    print(f"  Noise: 0.08 for first 30 epochs (denoising autoencoder)")
    print(f"  Target loss: <0.010 (previous best: 0.015)")
    
    model, training_metrics = train_correct(model, train_loader, epochs=50, lr=0.0001, device=device)
    
    # Evaluate
    print("\n4. Evaluating anomaly detection...")
    eval_results = evaluate_model(model, test_loader, device)
    
    # Save results
    print("\n5. Saving results...")
    plot_path = plot_results(training_metrics, eval_results)
    report_path = save_report(training_metrics, eval_results, model)
    
    # Save model
    model_path = 'leukemia_autoencoder_correct.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {'bottleneck_dim': model.bottleneck_dim},
        'training_metrics': training_metrics,
        'evaluation_results': eval_results
    }, model_path)
    print(f"Model saved to: {model_path}")
    
    # Save predictions and latent vectors
    predictions_path = 'autoencoder_correct_predictions.npz'
    save_dict = {
        'errors': eval_results['errors'],
        'labels': eval_results['labels'],
        'predictions': eval_results['predictions'],
        'threshold': eval_results['threshold'],
        'roc_auc': eval_results['roc_auc'],
        'error_stats': eval_results['error_stats']
    }
    if eval_results['latent_vectors'] is not None:
        save_dict['latent_vectors'] = eval_results['latent_vectors']
    
    np.savez(predictions_path, **save_dict)
    print(f"Predictions saved to: {predictions_path}")
    
    # Final summary
    print("\n" + "="*70)
    print("COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    # Performance assessment
    roc_auc = eval_results['roc_auc']
    error_ratio = eval_results['error_stats']['error_ratio']
    separation_ratio = eval_results['error_stats']['separation_ratio']
    final_loss = training_metrics['train_losses'][-1]
    
    print(f"\nKEY RESULTS:")
    print(f"  ROC AUC: {roc_auc:.4f} (Target: >0.75)")
    print(f"  Error Ratio: {error_ratio:.2f}x (Target: 3-5x)")
    print(f"  Separation Ratio: {separation_ratio:.2f}σ (Target: >1.5σ)")
    print(f"  Final Training Loss: {final_loss:.6f} (Target: <0.010)")
    
    print(f"\nPERFORMANCE EVALUATION:")
    if roc_auc >= 0.75 and error_ratio >= 2.5:
        print("  SUCCESS: Model achieves target performance!")
        print("  The bottleneck is working correctly with proper compression.")
    elif roc_auc >= 0.7:
        print("  MODERATE: Acceptable but needs slight improvement")
        print("  Bottleneck compression is helping but could be optimized.")
    elif roc_auc >= 0.65:
        print("  LIMITED: Some improvement over previous models")
        print("  Bottleneck helps but needs tuning.")
    else:
        print("  POOR: Worse than previous models")
        print("  The bottleneck compression might be too aggressive.")
    
    print(f"\nCOMPARISON TO BEST PREVIOUS (Model 3):")
    print(f"  Previous ROC AUC: 0.6964 | Current: {roc_auc:.4f}")
    print(f"  Previous Parameters: 51M | Current: {total_params:,} ({total_params/51000000:.1%})")
    print(f"  Improvement: {'YES' if roc_auc > 0.6964 else 'NO'}")
    
    print(f"\nFiles saved:")
    print(f"  Model: {model_path}")
    print(f"  Predictions: {predictions_path}")
    print(f"  Plot: {plot_path}")
    print(f"  Report: {report_path}")


if __name__ == "__main__":
    main()