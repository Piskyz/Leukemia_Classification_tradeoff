"""OPTUNA-OPTIMIZED AUTOENCODER FOR LEUKEMIA DETECTION
Training only - no validation during training
Post-training evaluation with Optuna-optimized hyperparameters
Best ROC AUC: 0.7472 | Parameters: 194,881
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
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import cv2
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import data loading functions
try:
    from Creador_labels import cargar_todos_datasets_con_labels
except ImportError:
    print("Warning: Creador_labels not found.")

try:
    from Carga_imagenes import cargar_training_all_original, cargar_training_hem_original
except ImportError:
    print("Warning: Carga_imagenes not found.")

# ============ AUTOENCODER ARCHITECTURE WITH OPTUNA OPTIMIZATIONS ============

class OptunaOptimizedAutoencoder(nn.Module):
    """
    Autoencoder with Optuna-optimized hyperparameters
    Based on trial #14: ROC AUC = 0.7472
    Channels multiplier: 0.5 | BatchNorm: True | Dropout: True (0.25)
    """
    def __init__(self):
        super(OptunaOptimizedAutoencoder, self).__init__()
        
        # OPTUNA OPTIMIZED HYPERPARAMETERS
        channels_multiplier = 0.5      # Optuna optimized
        use_batchnorm = True           # Optuna optimized
        use_dropout = True             # Optuna optimized
        dropout_rate = 0.25            # Optuna optimized
        
        # Calculate channels based on optimized multiplier
        base_channels = int(32 * channels_multiplier)  # 16 channels
        
        # ============ ENCODER ============
        encoder_layers = []
        
        # Layer 1: 450x450 -> 225x225
        encoder_layers.append(nn.Conv2d(1, base_channels, kernel_size=3, padding=1))
        if use_batchnorm:
            encoder_layers.append(nn.BatchNorm2d(base_channels))
        encoder_layers.append(nn.ReLU(True))
        encoder_layers.append(nn.MaxPool2d(2, stride=2))
        if use_dropout:
            encoder_layers.append(nn.Dropout2d(dropout_rate))
        
        # Layer 2: 225x225 -> 112x112
        encoder_layers.append(nn.Conv2d(base_channels, base_channels*2, kernel_size=3, padding=1))
        if use_batchnorm:
            encoder_layers.append(nn.BatchNorm2d(base_channels*2))
        encoder_layers.append(nn.ReLU(True))
        encoder_layers.append(nn.MaxPool2d(2, stride=2))
        if use_dropout:
            encoder_layers.append(nn.Dropout2d(dropout_rate))
        
        # Layer 3: 112x112 -> 56x56
        encoder_layers.append(nn.Conv2d(base_channels*2, base_channels*4, kernel_size=3, padding=1))
        if use_batchnorm:
            encoder_layers.append(nn.BatchNorm2d(base_channels*4))
        encoder_layers.append(nn.ReLU(True))
        encoder_layers.append(nn.MaxPool2d(2, stride=2))
        
        # Layer 4: 56x56 -> 28x28
        encoder_layers.append(nn.Conv2d(base_channels*4, base_channels*8, kernel_size=3, padding=1))
        if use_batchnorm:
            encoder_layers.append(nn.BatchNorm2d(base_channels*8))
        encoder_layers.append(nn.ReLU(True))
        encoder_layers.append(nn.MaxPool2d(2, stride=2))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # ============ DECODER ============
        decoder_layers = []
        
        # Layer 1: 28x28 -> 56x56
        decoder_layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        decoder_layers.append(nn.Conv2d(base_channels*8, base_channels*4, kernel_size=3, padding=1))
        if use_batchnorm:
            decoder_layers.append(nn.BatchNorm2d(base_channels*4))
        decoder_layers.append(nn.ReLU(True))
        
        # Layer 2: 56x56 -> 112x112
        decoder_layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        decoder_layers.append(nn.Conv2d(base_channels*4, base_channels*2, kernel_size=3, padding=1))
        if use_batchnorm:
            decoder_layers.append(nn.BatchNorm2d(base_channels*2))
        decoder_layers.append(nn.ReLU(True))
        
        # Layer 3: 112x112 -> 225x225
        decoder_layers.append(nn.Upsample(size=(225, 225), mode='bilinear', align_corners=True))
        decoder_layers.append(nn.Conv2d(base_channels*2, base_channels, kernel_size=3, padding=1))
        if use_batchnorm:
            decoder_layers.append(nn.BatchNorm2d(base_channels))
        decoder_layers.append(nn.ReLU(True))
        
        # Layer 4: 225x225 -> 450x450
        decoder_layers.append(nn.Upsample(size=(450, 450), mode='bilinear', align_corners=True))
        decoder_layers.append(nn.Conv2d(base_channels, 1, kernel_size=3, padding=1))
        decoder_layers.append(nn.Sigmoid())
        
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# ============ DATA PREPARATION FUNCTIONS ============

def convert_to_grayscale(images):
    """Convert images to grayscale"""
    processed = []
    for img in images:
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        processed.append(np.expand_dims(gray, axis=-1))
    return np.array(processed)

def prepare_optuna_data(datasets, batch_size=16):
    """Prepare data for anomaly detection training with Optuna splits"""
    print("Preparing data for Optuna-optimized training...")
    
    # Extract healthy (hem) and leukemia (all) images
    hem_images = []
    for key in datasets:
        if 'hem' in key.lower():
            hem_images.extend(datasets[key])
            
    all_images = []
    for key in datasets:
        if 'all' in key.lower():
            all_images.extend(datasets[key])
    
    # Convert to grayscale
    hem_gray = convert_to_grayscale(hem_images)
    all_gray = convert_to_grayscale(all_images)
    
    # Normalize to [0, 1]
    hem_gray = hem_gray.astype('float32') / 255.0
    all_gray = all_gray.astype('float32') / 255.0
    
    # Split healthy: 60% train, 20% test (like Optuna)
    X_healthy_temp, X_healthy_test = train_test_split(hem_gray, test_size=0.2, random_state=42)
    X_healthy_train, X_healthy_val = train_test_split(X_healthy_temp, test_size=0.25, random_state=42)
    
    # Split leukemia: 50% test (like Optuna validation split)
    X_leukemia_val, X_leukemia_test = train_test_split(all_gray, test_size=0.5, random_state=42)
    
    # Take balanced amounts for test (mimicking Optuna validation)
    n_test = min(len(X_healthy_test), len(X_leukemia_test))
    
    # Training: healthy only
    X_train = X_healthy_train
    
    # Test: balanced healthy + leukemia
    X_test = np.concatenate([X_healthy_test[:n_test], X_leukemia_test[:n_test]], axis=0)
    y_test = np.concatenate([np.zeros(n_test), np.ones(n_test)], axis=0)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).permute(0, 3, 1, 2)
    X_test_tensor = torch.FloatTensor(X_test).permute(0, 3, 1, 2)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Create datasets and loaders
    train_dataset = TensorDataset(X_train_tensor, X_train_tensor)  # Autoencoder: input = target
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"  Training (Healthy only): {len(X_train_tensor)} images")
    print(f"  Test (Balanced Healthy + Leukemia): {len(X_test_tensor)} images")
    print(f"    - Healthy in test: {n_test}")
    print(f"    - Leukemia in test: {n_test}")
    
    return train_loader, test_loader, (X_train_tensor, X_test_tensor, y_test_tensor)

# ============ TRAINING FUNCTIONS ============

def train_optuna_autoencoder(model, train_loader, epochs=50, device='cpu'):
    """Train autoencoder with Optuna-optimized parameters - only training loss"""
    
    # OPTUNA OPTIMIZED TRAINING HYPERPARAMETERS
    learning_rate = 0.0031404382679661026    # Optuna optimized
    weight_decay = 0.00011554231483529049    # Optuna optimized
    noise_level = 0.125                       # Optuna optimized
    
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)  # Optuna optimized
    
    train_losses = []
    epoch_times = []
    
    print(f"\nTraining Optuna-Optimized Autoencoder on {device}...")
    print(f"Optimizer: AdamW | LR: {learning_rate:.6f} | Weight decay: {weight_decay:.6f}")
    print(f"Noise level: {noise_level:.3f} (for first {epochs//2} epochs)")
    print(f"{'Epoch':^6} | {'Train Loss':^12} | {'Time (s)':^10}")
    print("-" * 35)
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        model.train()
        total_loss = 0.0
        
        for data, _ in train_loader:
            data = data.to(device)
            
            # Add noise for first half of training (Optuna optimized)
            if noise_level > 0 and epoch < epochs // 2:
                noise = torch.randn_like(data) * noise_level
                noisy_data = torch.clamp(data + noise, 0, 1)
            else:
                noisy_data = data
            
            optimizer.zero_grad()
            reconstruction = model(noisy_data)
            loss = criterion(reconstruction, data)
            loss.backward()
            
            # Clip gradients for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - epoch_start
        
        train_losses.append(avg_loss)
        epoch_times.append(epoch_time)
        
        print(f"{epoch+1:^6} | {avg_loss:^12.6f} | {epoch_time:^10.2f}")
    
    return model, {'train_losses': train_losses, 'epoch_times': epoch_times}

# ============ EVALUATION FUNCTIONS ============

def evaluate_optuna_autoencoder(model, test_loader, device='cpu'):
    """Evaluate autoencoder after training"""
    
    model.eval()
    criterion = nn.MSELoss(reduction='none')
    
    all_errors = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            
            reconstruction = model(data)
            loss = criterion(reconstruction, data)
            
            # Calculate mean error per image
            batch_errors = loss.mean(dim=(1, 2, 3)).cpu().numpy()
            
            all_errors.extend(batch_errors)
            all_labels.extend(labels.numpy())
    
    return np.array(all_errors), np.array(all_labels)

def analyze_optuna_results(errors, true_labels, model):
    """Complete analysis of results with Optuna comparison"""
    
    # Separate errors by class
    healthy_errors = errors[true_labels == 0]
    leukemia_errors = errors[true_labels == 1]
    
    print(f"\n{'='*70}")
    print("POST-TRAINING ANALYSIS WITH OPTUNA COMPARISON")
    print(f"{'='*70}")
    
    print(f"\nError Statistics:")
    print(f"Healthy samples (n={len(healthy_errors)}):")
    print(f"  Mean MSE: {np.mean(healthy_errors):.6f}")
    print(f"  Std MSE:  {np.std(healthy_errors):.6f}")
    print(f"  Min:      {np.min(healthy_errors):.6f}")
    print(f"  Max:      {np.max(healthy_errors):.6f}")
    
    print(f"\nLeukemia samples (n={len(leukemia_errors)}):")
    print(f"  Mean MSE: {np.mean(leukemia_errors):.6f}")
    print(f"  Std MSE:  {np.std(leukemia_errors):.6f}")
    print(f"  Min:      {np.min(leukemia_errors):.6f}")
    print(f"  Max:      {np.max(leukemia_errors):.6f}")
    
    # Calculate separation metrics
    separation = np.mean(leukemia_errors) - np.mean(healthy_errors)
    separation_ratio = separation / np.std(healthy_errors) if np.std(healthy_errors) > 0 else 0
    error_ratio = np.mean(leukemia_errors) / np.mean(healthy_errors)
    
    print(f"\nSeparation Analysis:")
    print(f"  Difference: {separation:.6f}")
    print(f"  Separation ratio: {separation_ratio:.2f}σ")
    print(f"  Error ratio: {error_ratio:.2f}x")
    
    # Find optimal threshold using ROC
    fpr, tpr, thresholds = roc_curve(true_labels, errors)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    roc_auc = auc(fpr, tpr)
    
    # Calculate ROC AUC using sklearn
    roc_auc_score_val = roc_auc_score(true_labels, errors)
    
    # Percentile-based thresholds
    percentile_95 = np.percentile(healthy_errors, 95)
    percentile_99 = np.percentile(healthy_errors, 99)
    
    print(f"\nThreshold Analysis:")
    print(f"ROC AUC (sklearn): {roc_auc_score_val:.4f}")
    print(f"ROC AUC (manual): {roc_auc:.4f}")
    print(f"Optimal threshold (Youden's J): {optimal_threshold:.6f}")
    print(f"95th percentile (healthy): {percentile_95:.6f}")
    print(f"99th percentile (healthy): {percentile_99:.6f}")
    
    # Test different thresholds
    thresholds_to_test = {
        'optimal': optimal_threshold,
        'p95': percentile_95,
        'p99': percentile_99
    }
    
    best_results = None
    best_f1 = 0
    
    for thresh_name, threshold in thresholds_to_test.items():
        predictions = (errors > threshold).astype(int)
        
        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / len(true_labels)
        
        print(f"\nThreshold: {thresh_name} ({threshold:.6f})")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-score:  {f1:.4f}")
        print(f"  Confusion Matrix:")
        print(f"    TN: {tn}, FP: {fp}")
        print(f"    FN: {fn}, TP: {tp}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_results = {
                'threshold_name': thresh_name,
                'threshold': threshold,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc_score_val,
                'confusion_matrix': [[tn, fp], [fn, tp]],
                'errors': errors,
                'true_labels': true_labels,
                'healthy_mean': float(np.mean(healthy_errors)),
                'leukemia_mean': float(np.mean(leukemia_errors)),
                'separation': float(separation),
                'separation_ratio': float(separation_ratio),
                'error_ratio': float(error_ratio)
            }
    
    # Get model parameter count
    total_params = sum(p.numel() for p in model.parameters())
    
    # Comparison with Optuna results
    print(f"\n{'='*70}")
    print("COMPARISON WITH OPTUNA OPTIMIZATION")
    print(f"{'='*70}")
    print(f"Optuna Trial #14 Results:")
    print(f"  Validation ROC AUC: 0.7472")
    print(f"  Parameters: 194,881")
    print(f"\nCurrent Training Results:")
    print(f"  Test ROC AUC: {roc_auc_score_val:.4f}")
    print(f"  Parameters: {total_params:,}")
    
    if roc_auc_score_val > 0.7472:
        improvement = roc_auc_score_val - 0.7472
        print(f"\nSUCCESS: Current model outperforms Optuna by {improvement:.4f} ({improvement/0.7472*100:.1f}%)")
    elif roc_auc_score_val < 0.7472:
        difference = 0.7472 - roc_auc_score_val
        print(f"\nNote: Optuna performed better by {difference:.4f}")
    else:
        print(f"\nNote: Performance matches Optuna results")
    
    # Detailed classification report for best threshold
    print(f"\n{'='*70}")
    print(f"BEST PERFORMANCE: {best_results['threshold_name'].upper()} THRESHOLD")
    print(f"{'='*70}")
    
    best_predictions = (errors > best_results['threshold']).astype(int)
    print("\nClassification Report:")
    print(classification_report(true_labels, best_predictions, 
                                target_names=['Healthy', 'Leukemia']))
    
    return best_results, errors, true_labels

def plot_optuna_results(train_metrics, errors, true_labels, best_results, save_dir="optuna_autoencoder_results"):
    """Create visualization plots for Optuna-optimized model"""
    
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Training loss
    axes[0, 0].plot(train_metrics['train_losses'], 'b-', linewidth=2)
    axes[0, 0].set_title('Training Loss (MSE)')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Error distributions
    healthy_errors = errors[true_labels == 0]
    leukemia_errors = errors[true_labels == 1]
    
    axes[0, 1].hist(healthy_errors, bins=50, alpha=0.7, label='Healthy', density=True, color='green')
    axes[0, 1].hist(leukemia_errors, bins=50, alpha=0.7, label='Leukemia', density=True, color='red')
    axes[0, 1].axvline(best_results['threshold'], color='black', linestyle='--', 
                       label=f"Threshold: {best_results['threshold']:.6f}")
    axes[0, 1].set_title('Error Distributions by Class')
    axes[0, 1].set_xlabel('Reconstruction Error (MSE)')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: ROC Curve
    fpr, tpr, _ = roc_curve(true_labels, errors)
    roc_auc = auc(fpr, tpr)
    
    axes[0, 2].plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})', linewidth=2)
    axes[0, 2].plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.5)
    axes[0, 2].set_title('ROC Curve')
    axes[0, 2].set_xlabel('False Positive Rate')
    axes[0, 2].set_ylabel('True Positive Rate')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Confusion Matrix
    cm = best_results['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Pred Healthy', 'Pred Leukemia'],
                yticklabels=['True Healthy', 'True Leukemia'],
                ax=axes[1, 0])
    axes[1, 0].set_title(f'Confusion Matrix\n({best_results["threshold_name"]} threshold)')
    
    # Plot 5: Metrics comparison
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    metrics_values = [
        best_results['accuracy'],
        best_results['precision'],
        best_results['recall'],
        best_results['f1']
    ]
    
    colors = ['blue', 'green', 'orange', 'red']
    bars = axes[1, 1].bar(metrics_names, metrics_values, color=colors)
    axes[1, 1].set_title('Performance Metrics')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, metrics_values):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 6: Training time per epoch
    axes[1, 2].plot(train_metrics['epoch_times'], 'g-', linewidth=2)
    axes[1, 2].set_title('Training Time per Epoch')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Time (seconds)')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.suptitle(f'Optuna-Optimized Autoencoder Results - {timestamp}', fontsize=16)
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(save_dir, f'optuna_results_{timestamp}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlots saved to: {plot_path}")
    
    # Save raw data
    data_path = os.path.join(save_dir, f'optuna_results_{timestamp}.npz')
    np.savez(data_path,
             train_losses=train_metrics['train_losses'],
             epoch_times=train_metrics['epoch_times'],
             errors=errors,
             true_labels=true_labels,
             best_threshold=best_results['threshold'],
             best_accuracy=best_results['accuracy'],
             best_recall=best_results['recall'],
             best_precision=best_results['precision'],
             best_f1=best_results['f1'],
             roc_auc=best_results['roc_auc'])
    
    print(f"Data saved to: {data_path}")
    
    return plot_path

# ============ MAIN FUNCTION ============

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("\n" + "="*70)
    print("OPTUNA-OPTIMIZED AUTOENCODER FOR LEUKEMIA DETECTION")
    print("Based on Trial #14: ROC AUC = 0.7472")
    print("Parameters: 194,881")
    print("="*70)
    
    # 1. Load data
    print("\n1. Loading datasets...")
    
    # Try to load data using your functions
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
    
    print(f"Loaded {len([img for img_list in datasets.values() for img in img_list])} total images")
    
    # 2. Prepare data (Optuna-style splits)
    train_loader, test_loader, _ = prepare_optuna_data(datasets, batch_size=16)
    
    # 3. Create Optuna-optimized model
    print("\n2. Creating Optuna-optimized autoencoder...")
    model = OptunaOptimizedAutoencoder()
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"\nModel Architecture (Optuna-optimized):")
    print(f"  Base channels: 16 (32 * 0.5 multiplier)")
    print(f"  BatchNorm: Enabled")
    print(f"  Dropout: Enabled (rate: 0.25)")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Parameter reduction from original: {(776833 - total_params)/776833*100:.1f}%")
    
    # 4. Train (only training, no validation)
    print("\n" + "="*70)
    print("3. TRAINING PHASE (NO VALIDATION DURING TRAINING)")
    print("="*70)
    
    model, train_metrics = train_optuna_autoencoder(
        model=model,
        train_loader=train_loader,
        epochs=50,  # Increased from Optuna's 14 for better training
        device=device
    )
    
    # 5. Evaluate after training
    print("\n" + "="*70)
    print("4. POST-TRAINING EVALUATION")
    print("="*70)
    
    errors, true_labels = evaluate_optuna_autoencoder(model, test_loader, device)
    
    # 6. Analyze results with Optuna comparison
    best_results, errors, true_labels = analyze_optuna_results(errors, true_labels, model)
    
    # 7. Plot results
    plot_path = plot_optuna_results(train_metrics, errors, true_labels, best_results)
    
    # 8. Save model
    model_path = 'leukemia_autoencoder_optuna_optimized.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_metrics': train_metrics,
        'best_threshold': best_results['threshold'],
        'best_results': best_results,
        'optuna_comparison': {
            'optuna_roc_auc': 0.7472,
            'current_roc_auc': best_results['roc_auc'],
            'optuna_params': 194881,
            'current_params': total_params,
            'improvement': best_results['roc_auc'] - 0.7472
        }
    }, model_path)
    
    print(f"\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Model saved as: {model_path}")
    print(f"Best threshold: {best_results['threshold_name']} = {best_results['threshold']:.6f}")
    print(f"Best F1-score: {best_results['f1']:.4f}")
    print(f"Recall (Leukemia detection): {best_results['recall']:.4f}")
    print(f"ROC AUC: {best_results['roc_auc']:.4f}")
    
    print(f"\nFinal Comparison:")
    print(f"  Optuna validation AUC: 0.7472")
    print(f"  Current test AUC: {best_results['roc_auc']:.4f}")
    if best_results['roc_auc'] > 0.7472:
        print(f"  Improvement: +{best_results['roc_auc'] - 0.7472:.4f}")
    print(f"  Parameter reduction: {(776833 - total_params)/776833*100:.1f}%")

if __name__ == "__main__":
    main()