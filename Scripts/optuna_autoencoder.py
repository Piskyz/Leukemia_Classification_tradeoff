"""OPTUNA HYPERPARAMETER OPTIMIZATION FOR LEUKEMIA AUTOENCODER
Finds optimal hyperparameters without training final model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import os
import optuna
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import cv2
import warnings
import json
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


class OriginalAutoencoder(nn.Module):
    """
    Original Autoencoder Architecture (from Model 1)
    This will be the base for hyperparameter optimization
    """
    def __init__(self, 
                 channels_multiplier=1.0,
                 use_batchnorm=True,
                 use_dropout=True,
                 dropout_rate=0.1):
        super(OriginalAutoencoder, self).__init__()
        
        # Calculate channels based on multiplier
        base_channels = int(32 * channels_multiplier)
        
        # ============ ENCODER ============
        encoder_layers = []
        
        # Layer 1
        encoder_layers.append(nn.Conv2d(1, base_channels, kernel_size=3, padding=1))
        if use_batchnorm:
            encoder_layers.append(nn.BatchNorm2d(base_channels))
        encoder_layers.append(nn.ReLU(True))
        encoder_layers.append(nn.MaxPool2d(2, stride=2))
        if use_dropout:
            encoder_layers.append(nn.Dropout2d(dropout_rate))
        
        # Layer 2
        encoder_layers.append(nn.Conv2d(base_channels, base_channels*2, kernel_size=3, padding=1))
        if use_batchnorm:
            encoder_layers.append(nn.BatchNorm2d(base_channels*2))
        encoder_layers.append(nn.ReLU(True))
        encoder_layers.append(nn.MaxPool2d(2, stride=2))
        if use_dropout:
            encoder_layers.append(nn.Dropout2d(dropout_rate))
        
        # Layer 3
        encoder_layers.append(nn.Conv2d(base_channels*2, base_channels*4, kernel_size=3, padding=1))
        if use_batchnorm:
            encoder_layers.append(nn.BatchNorm2d(base_channels*4))
        encoder_layers.append(nn.ReLU(True))
        encoder_layers.append(nn.MaxPool2d(2, stride=2))
        
        # Layer 4
        encoder_layers.append(nn.Conv2d(base_channels*4, base_channels*8, kernel_size=3, padding=1))
        if use_batchnorm:
            encoder_layers.append(nn.BatchNorm2d(base_channels*8))
        encoder_layers.append(nn.ReLU(True))
        encoder_layers.append(nn.MaxPool2d(2, stride=2))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # ============ DECODER ============
        decoder_layers = []
        
        # Layer 1
        decoder_layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        decoder_layers.append(nn.Conv2d(base_channels*8, base_channels*4, kernel_size=3, padding=1))
        if use_batchnorm:
            decoder_layers.append(nn.BatchNorm2d(base_channels*4))
        decoder_layers.append(nn.ReLU(True))
        
        # Layer 2
        decoder_layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        decoder_layers.append(nn.Conv2d(base_channels*4, base_channels*2, kernel_size=3, padding=1))
        if use_batchnorm:
            decoder_layers.append(nn.BatchNorm2d(base_channels*2))
        decoder_layers.append(nn.ReLU(True))
        
        # Layer 3
        decoder_layers.append(nn.Upsample(size=(225, 225), mode='bilinear', align_corners=True))
        decoder_layers.append(nn.Conv2d(base_channels*2, base_channels, kernel_size=3, padding=1))
        if use_batchnorm:
            decoder_layers.append(nn.BatchNorm2d(base_channels))
        decoder_layers.append(nn.ReLU(True))
        
        # Layer 4
        decoder_layers.append(nn.Upsample(size=(450, 450), mode='bilinear', align_corners=True))
        decoder_layers.append(nn.Conv2d(base_channels, 1, kernel_size=3, padding=1))
        decoder_layers.append(nn.Sigmoid())
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Store hyperparameters
        self.channels_multiplier = channels_multiplier
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
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


def prepare_optuna_data(datasets, batch_size=16):
    """
    Prepare data for Optuna optimization with train/val/test splits.
    """
    print("\nPreparing data for Optuna optimization...")
    
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
    
    # Split healthy: 60% train, 20% validation, 20% test
    X_healthy_temp, X_healthy_test = train_test_split(hem_gray, test_size=0.2, random_state=42)
    X_healthy_train, X_healthy_val = train_test_split(X_healthy_temp, test_size=0.25, random_state=42)
    
    # Split leukemia: 50% validation, 50% test (none for training)
    X_leukemia_val, X_leukemia_test = train_test_split(all_gray, test_size=0.5, random_state=42)
    
    # Take balanced amounts for validation
    n_val = min(len(X_healthy_val), len(X_leukemia_val))
    n_test = min(len(X_healthy_test), len(X_leukemia_test))
    
    # Final datasets - for Optuna we only need train and val
    X_train = X_healthy_train
    X_val = np.concatenate([X_healthy_val[:n_val], X_leukemia_val[:n_val]], axis=0)
    
    y_val = np.concatenate([np.zeros(n_val), np.ones(n_val)], axis=0)
    
    # To tensors
    X_train_tensor = torch.FloatTensor(X_train).permute(0, 3, 1, 2)
    X_val_tensor = torch.FloatTensor(X_val).permute(0, 3, 1, 2)
    
    y_val_tensor = torch.LongTensor(y_val)
    
    # Datasets
    train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\nData split for Optuna:")
    print(f"  Train (healthy only): {len(X_train)} images")
    print(f"  Validation (balanced): {len(X_val)} images ({n_val} healthy, {n_val} leukemia)")
    
    return train_loader, val_loader, y_val


def train_autoencoder_optuna(model, train_loader, val_loader, hyperparams, device='cuda', timeout_seconds=210):
    """
    Train autoencoder for Optuna optimization with timeout.
    Returns validation ROC AUC.
    """
    start_time = time.time()
    
    try:
        model = model.to(device)
        
        criterion = nn.MSELoss()
        
        # Get optimizer parameters from hyperparams
        optimizer_name = hyperparams['optimizer']
        lr = hyperparams['learning_rate']
        weight_decay = hyperparams['weight_decay']
        
        if optimizer_name == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:  # sgd
            optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        
        epochs = hyperparams['epochs']
        noise_level = hyperparams['noise_level']
        
        # Reduce epochs if we're running out of time
        max_allowed_epochs = min(epochs, 15)  # Cap at 15 epochs for timeout
        actual_epochs = max_allowed_epochs
        
        # Training loop
        for epoch in range(actual_epochs):
            # Check timeout
            if time.time() - start_time > timeout_seconds * 0.8:  # Stop early at 80% of timeout
                print(f"  Timeout warning: Stopping early at epoch {epoch+1}/{actual_epochs}")
                break
                
            model.train()
            
            for batch_idx, (data, target) in enumerate(train_loader):
                # Check timeout more frequently
                if time.time() - start_time > timeout_seconds * 0.8:
                    break
                    
                data, target = data.to(device), target.to(device)
                
                # Add noise if specified
                if noise_level > 0 and epoch < actual_epochs // 2:
                    noise = torch.randn_like(data) * noise_level
                    noisy_data = torch.clamp(data + noise, 0, 1)
                else:
                    noisy_data = data
                
                optimizer.zero_grad()
                output = model(noisy_data)
                loss = criterion(output, target)
                loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
        
        # Evaluate on validation set
        model.eval()
        val_errors = []
        val_labels = []
        
        with torch.no_grad():
            for data, labels in val_loader:
                data = data.to(device)
                recon = model(data)
                
                # MSE per image
                mse = torch.mean((recon - data) ** 2, dim=(1, 2, 3))
                
                val_errors.extend(mse.cpu().numpy())
                val_labels.extend(labels.numpy())
        
        val_errors = np.array(val_errors)
        val_labels = np.array(val_labels)
        
        # Calculate ROC AUC
        if len(np.unique(val_labels)) > 1:
            roc_auc = roc_auc_score(val_labels, val_errors)
        else:
            roc_auc = 0.5
        
        training_time = time.time() - start_time
        print(f"  Trial completed in {training_time:.1f}s, ROC AUC: {roc_auc:.4f}")
        
        return roc_auc
        
    except Exception as e:
        print(f"Training failed: {e}")
        raise


def objective(trial, train_loader, val_loader, device, timeout_per_trial=210):
    """
    Optuna objective function to maximize ROC AUC with timeout.
    """
    start_time = time.time()
    
    # Suggest hyperparameters
    # First suggest whether to use dropout
    use_dropout = trial.suggest_categorical('use_dropout', [True, False])
    
    # Then suggest dropout rate only if dropout is used
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.3, step=0.05) if use_dropout else 0.0
    
    # Suggest other hyperparameters
    channels_multiplier = trial.suggest_float('channels_multiplier', 0.5, 2.0, step=0.25)
    use_batchnorm = trial.suggest_categorical('use_batchnorm', [True, False])
    
    # Create hyperparams dictionary
    hyperparams = {
        'channels_multiplier': channels_multiplier,
        'use_batchnorm': use_batchnorm,
        'use_dropout': use_dropout,
        'dropout_rate': dropout_rate,
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
        'optimizer': trial.suggest_categorical('optimizer', ['adam', 'adamw']),
        'epochs': trial.suggest_int('epochs', 10, 20),  # Increased for 3.5 minutes
        'noise_level': trial.suggest_float('noise_level', 0.0, 0.15, step=0.025),
    }
    
    # Create model with suggested hyperparameters
    model = OriginalAutoencoder(
        channels_multiplier=hyperparams['channels_multiplier'],
        use_batchnorm=hyperparams['use_batchnorm'],
        use_dropout=hyperparams['use_dropout'],
        dropout_rate=hyperparams['dropout_rate']
    )
    
    # Train and evaluate
    try:
        roc_auc = train_autoencoder_optuna(model, train_loader, val_loader, hyperparams, device, timeout_seconds=timeout_per_trial*0.9)
        
        # Calculate number of parameters
        total_params = sum(p.numel() for p in model.parameters())
        
        # Add parameter count as intermediate value
        trial.set_user_attr('total_params', total_params)
        trial.set_user_attr('hyperparams', hyperparams)
        trial.set_user_attr('training_time', time.time() - start_time)
        
        return roc_auc
        
    except Exception as e:
        print(f"Trial failed with error: {e}")
        # Return a low score for failed trials
        return 0.1


def run_optuna_optimization(train_loader, val_loader, device, n_trials=20, timeout_per_trial=210):
    """
    Run Optuna hyperparameter optimization with timeout.
    Returns the best hyperparameters.
    """
    print(f"\nStarting Optuna hyperparameter optimization ({n_trials} trials)...")
    print(f"Device: {device}")
    print(f"Timeout per trial: {timeout_per_trial} seconds ({timeout_per_trial/60:.1f} minutes)")
    print(f"Estimated total time: {(timeout_per_trial * n_trials)/60:.1f} minutes")
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        study_name='leukemia_autoencoder_optuna',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # Run optimization with timeout
    study.optimize(
        lambda trial: objective(trial, train_loader, val_loader, device, timeout_per_trial),
        n_trials=n_trials,
        n_jobs=1,  # Single job for timeout control
        show_progress_bar=True,
        timeout=timeout_per_trial * n_trials * 1.5  # Total timeout with buffer
    )
    
    # Print trial statistics
    print(f"\nTrial Statistics:")
    print(f"  Completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"  Failed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")
    print(f"  Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    
    return study


def save_hyperparameters(study, output_dir='optuna_results'):
    """
    Save the best hyperparameters to a JSON file.
    """
    if len(study.trials) == 0:
        print("No trials completed successfully!")
        return None
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get best hyperparameters
    best_trial = study.best_trial
    best_params = best_trial.params
    best_value = best_trial.value
    total_params = best_trial.user_attrs.get('total_params', 0)
    training_time = best_trial.user_attrs.get('training_time', 0)
    
    # Add metadata
    hyperparameters = {
        'best_roc_auc': float(best_value),
        'total_parameters': int(total_params),
        'trial_number': int(best_trial.number),
        'training_time_seconds': float(training_time),
        'hyperparameters': best_params,
        'architecture': {
            'channels_multiplier': float(best_params['channels_multiplier']),
            'use_batchnorm': bool(best_params['use_batchnorm']),
            'use_dropout': bool(best_params['use_dropout']),
            'dropout_rate': float(best_params.get('dropout_rate', 0.0))
        },
        'training': {
            'learning_rate': float(best_params['learning_rate']),
            'weight_decay': float(best_params['weight_decay']),
            'optimizer': str(best_params['optimizer']),
            'epochs': int(best_params['epochs']),
            'noise_level': float(best_params['noise_level'])
        },
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'n_trials': len(study.trials),
        'n_completed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    }
    
    # Save to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(output_dir, f'best_hyperparameters_{timestamp}.json')
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(hyperparameters, f, indent=4, ensure_ascii=False)
    
    print(f"\nBest hyperparameters saved to: {json_path}")
    
    # Also save a simple version for easy import
    simple_path = os.path.join(output_dir, 'best_hyperparameters.json')
    with open(simple_path, 'w', encoding='utf-8') as f:
        json.dump(hyperparameters, f, indent=4, ensure_ascii=False)
    
    return hyperparameters, json_path


def print_optimization_results(study, hyperparameters):
    """
    Print optimization results.
    """
    print("\n" + "="*70)
    print("OPTUNA HYPERPARAMETER OPTIMIZATION COMPLETE")
    print("="*70)
    
    if len(study.trials) == 0:
        print("No trials completed successfully!")
        return
    
    print(f"\nBest trial #{study.best_trial.number}:")
    print(f"  ROC AUC: {study.best_value:.4f}")
    print(f"  Total parameters: {study.best_trial.user_attrs.get('total_params', 'N/A'):,}")
    print(f"  Training time: {study.best_trial.user_attrs.get('training_time', 0):.1f}s")
    
    print(f"\n BEST HYPERPARAMETERS:")
    print(f"\nARCHITECTURE:")
    print(f"  Channels multiplier: {hyperparameters['architecture']['channels_multiplier']}")
    print(f"  Use BatchNorm: {hyperparameters['architecture']['use_batchnorm']}")
    print(f"  Use Dropout: {hyperparameters['architecture']['use_dropout']}")
    if hyperparameters['architecture']['use_dropout']:
        print(f"  Dropout rate: {hyperparameters['architecture']['dropout_rate']}")
    
    print(f"\nTRAINING:")
    print(f"  Learning rate: {hyperparameters['training']['learning_rate']:.6f}")
    print(f"  Weight decay: {hyperparameters['training']['weight_decay']:.6f}")
    print(f"  Optimizer: {hyperparameters['training']['optimizer']}")
    print(f"  Epochs: {hyperparameters['training']['epochs']}")
    print(f"  Noise level: {hyperparameters['training']['noise_level']:.3f}")
    
    print(f"\n OPTIMIZATION STATS:")
    print(f"  Total trials: {len(study.trials)}")
    print(f"  Completed trials: {hyperparameters['n_completed_trials']}")
    print(f"  Best ROC AUC: {hyperparameters['best_roc_auc']:.4f}")
    print(f"  Completed at: {hyperparameters['timestamp']}")
    
    # Show top 5 trials
    print(f"\n TOP 5 TRIALS:")
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    sorted_trials = sorted(completed_trials, key=lambda x: x.value, reverse=True)
    for i, trial in enumerate(sorted_trials[:5]):
        print(f"  {i+1}. Trial {trial.number}: ROC AUC = {trial.value:.4f}, "
              f"Params = {trial.user_attrs.get('total_params', 'N/A'):,}, "
              f"Time = {trial.user_attrs.get('training_time', 0):.1f}s")


def plot_optimization_history(study, output_dir='optuna_results'):
    """
    Plot optimization history and save to file.
    """
    if len(study.trials) == 0:
        return
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Optimization history
    ax1 = axes[0]
    
    # Get trial numbers and values
    trial_numbers = []
    values = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            trial_numbers.append(trial.number)
            values.append(trial.value)
    
    if trial_numbers:
        # Sort by trial number
        sorted_data = sorted(zip(trial_numbers, values))
        trial_numbers_sorted, values_sorted = zip(*sorted_data)
        
        # Plot optimization history
        ax1.plot(trial_numbers_sorted, values_sorted, 'b-', marker='o', markersize=4, linewidth=1.5)
        ax1.fill_between(trial_numbers_sorted, values_sorted, alpha=0.2)
        
        # Highlight best trial
        best_idx = values_sorted.index(max(values_sorted))
        ax1.plot(trial_numbers_sorted[best_idx], values_sorted[best_idx], 'ro', markersize=8, label=f'Best: {values_sorted[best_idx]:.4f}')
        
        ax1.set_xlabel('Trial Number')
        ax1.set_ylabel('ROC AUC')
        ax1.set_title('Optimization History')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim(0.4, 1.0)
    else:
        ax1.text(0.5, 0.5, 'No completed trials', ha='center', va='center')
        ax1.set_title('Optimization History')
    
    # Plot 2: Trial values distribution
    ax2 = axes[1]
    if values:
        ax2.hist(values, bins=10, alpha=0.7, color='green', edgecolor='black')
        ax2.axvline(np.mean(values), color='red', linestyle='--', label=f'Mean: {np.mean(values):.4f}')
        ax2.axvline(np.max(values), color='blue', linestyle='--', label=f'Best: {np.max(values):.4f}')
        ax2.set_xlabel('ROC AUC')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Trial Values Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No completed trials', ha='center', va='center')
        ax2.set_title('Trial Values Distribution')
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(output_dir, f'optuna_history_{timestamp}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Optimization plots saved to: {plot_path}")
    return plot_path


def create_hyperparameter_usage_example(hyperparameters, output_dir='optuna_results'):
    """
    Create a Python file showing how to use the hyperparameters.
    """
    example_code = f'''"""
HOW TO USE THE OPTUNA-OPTIMIZED HYPERPARAMETERS
Generated: {hyperparameters['timestamp']}
Best ROC AUC: {hyperparameters['best_roc_auc']:.4f}
"""

import torch
import torch.nn as nn

class OriginalAutoencoder(nn.Module):
    def __init__(self, 
                 channels_multiplier={hyperparameters['architecture']['channels_multiplier']},
                 use_batchnorm={hyperparameters['architecture']['use_batchnorm']},
                 use_dropout={hyperparameters['architecture']['use_dropout']},
                 dropout_rate={hyperparameters['architecture']['dropout_rate']}):
        super(OriginalAutoencoder, self).__init__()
        
        base_channels = int(32 * channels_multiplier)
        
        # ENCODER
        encoder_layers = []
        
        # Layer 1
        encoder_layers.append(nn.Conv2d(1, base_channels, kernel_size=3, padding=1))
        if use_batchnorm:
            encoder_layers.append(nn.BatchNorm2d(base_channels))
        encoder_layers.append(nn.ReLU(True))
        encoder_layers.append(nn.MaxPool2d(2, stride=2))
        if use_dropout:
            encoder_layers.append(nn.Dropout2d(dropout_rate))
        
        # Layer 2
        encoder_layers.append(nn.Conv2d(base_channels, base_channels*2, kernel_size=3, padding=1))
        if use_batchnorm:
            encoder_layers.append(nn.BatchNorm2d(base_channels*2))
        encoder_layers.append(nn.ReLU(True))
        encoder_layers.append(nn.MaxPool2d(2, stride=2))
        if use_dropout:
            encoder_layers.append(nn.Dropout2d(dropout_rate))
        
        # Layer 3
        encoder_layers.append(nn.Conv2d(base_channels*2, base_channels*4, kernel_size=3, padding=1))
        if use_batchnorm:
            encoder_layers.append(nn.BatchNorm2d(base_channels*4))
        encoder_layers.append(nn.ReLU(True))
        encoder_layers.append(nn.MaxPool2d(2, stride=2))
        
        # Layer 4
        encoder_layers.append(nn.Conv2d(base_channels*4, base_channels*8, kernel_size=3, padding=1))
        if use_batchnorm:
            encoder_layers.append(nn.BatchNorm2d(base_channels*8))
        encoder_layers.append(nn.ReLU(True))
        encoder_layers.append(nn.MaxPool2d(2, stride=2))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # DECODER
        decoder_layers = []
        
        decoder_layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        decoder_layers.append(nn.Conv2d(base_channels*8, base_channels*4, kernel_size=3, padding=1))
        if use_batchnorm:
            decoder_layers.append(nn.BatchNorm2d(base_channels*4))
        decoder_layers.append(nn.ReLU(True))
        
        decoder_layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        decoder_layers.append(nn.Conv2d(base_channels*4, base_channels*2, kernel_size=3, padding=1))
        if use_batchnorm:
            decoder_layers.append(nn.BatchNorm2d(base_channels*2))
        decoder_layers.append(nn.ReLU(True))
        
        decoder_layers.append(nn.Upsample(size=(225, 225), mode='bilinear', align_corners=True))
        decoder_layers.append(nn.Conv2d(base_channels*2, base_channels, kernel_size=3, padding=1))
        if use_batchnorm:
            decoder_layers.append(nn.BatchNorm2d(base_channels))
        decoder_layers.append(nn.ReLU(True))
        
        decoder_layers.append(nn.Upsample(size=(450, 450), mode='bilinear', align_corners=True))
        decoder_layers.append(nn.Conv2d(base_channels, 1, kernel_size=3, padding=1))
        decoder_layers.append(nn.Sigmoid())
        
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def get_optimized_hyperparameters():
    """
    Returns the Optuna-optimized hyperparameters.
    """
    hyperparams = {{
        # Architecture hyperparameters
        'channels_multiplier': {hyperparameters['architecture']['channels_multiplier']},
        'use_batchnorm': {hyperparameters['architecture']['use_batchnorm']},
        'use_dropout': {hyperparameters['architecture']['use_dropout']},
        'dropout_rate': {hyperparameters['architecture']['dropout_rate']},
        
        # Training hyperparameters
        'learning_rate': {hyperparameters['training']['learning_rate']},
        'weight_decay': {hyperparameters['training']['weight_decay']},
        'optimizer': '{hyperparameters['training']['optimizer']}',
        'epochs': {hyperparameters['training']['epochs']},
        'noise_level': {hyperparameters['training']['noise_level']},
    }}
    
    return hyperparams


def create_optimized_model():
    """
    Creates a model with optimized hyperparameters.
    """
    hyperparams = get_optimized_hyperparameters()
    
    model = OriginalAutoencoder(
        channels_multiplier=hyperparams['channels_multiplier'],
        use_batchnorm=hyperparams['use_batchnorm'],
        use_dropout=hyperparams['use_dropout'],
        dropout_rate=hyperparams['dropout_rate']
    )
    
    return model


if __name__ == "__main__":
    # Example usage
    hyperparams = get_optimized_hyperparameters()
    print("Optimized hyperparameters:")
    for key, value in hyperparams.items():
        print(f"  {{key}}: {{value}}")
    
    # Create model with optimized architecture
    model = create_optimized_model()
    print(f"\\nModel created with {{sum(p.numel() for p in model.parameters()):,}} parameters")
'''

    example_path = os.path.join(output_dir, 'use_optimized_hyperparameters.py')
    with open(example_path, 'w', encoding='utf-8') as f:
        f.write(example_code)
    
    print(f"Usage example saved to: {example_path}")
    return example_path


def main():
    """Main function for Optuna hyperparameter optimization."""
    print("\n" + "="*70)
    print("OPTUNA HYPERPARAMETER OPTIMIZATION")
    print("Finds optimal hyperparameters without final training")
    print("Trials: 20 | Timeout: 3.5 minutes (210 seconds) per trial")
    print("="*70)
    
    # Set random seeds for reproducibility
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
    
    # Prepare data for Optuna
    train_loader, val_loader, _ = prepare_optuna_data(datasets, batch_size=16)
    
    # Run Optuna optimization with 3.5 minute timeout per trial
    print("\n2. Running Optuna hyperparameter optimization...")
    n_trials = 20  # Increased to 20 trials
    timeout_per_trial = 210  # 3.5 minutes = 210 seconds
    study = run_optuna_optimization(train_loader, val_loader, device, 
                                    n_trials=n_trials, 
                                    timeout_per_trial=timeout_per_trial)
    
    # Check if any trials succeeded
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed_trials) == 0:
        print("\nERROR: No trials completed successfully!")
        print("Please check the following:")
        print("1. Data loading and preprocessing")
        print("2. Memory availability")
        print("3. Reduce hyperparameter search space")
        return None
    
    # Save hyperparameters
    print("\n3. Saving optimal hyperparameters...")
    hyperparameters, json_path = save_hyperparameters(study)
    
    # Print results
    print_optimization_results(study, hyperparameters)
    
    # Create plots
    print("\n4. Creating optimization plots...")
    plot_path = plot_optimization_history(study)
    
    # Create usage example
    print("\n5. Creating usage example...")
    example_path = create_hyperparameter_usage_example(hyperparameters)
    
    print(f"\n OPTIMIZATION COMPLETE!")
    print(f"\n Files created:")
    print(f"  Hyperparameters: {json_path}")
    print(f"  Simple copy: optuna_results/best_hyperparameters.json")
    print(f"  Optimization plots: {plot_path}")
    print(f"  Usage example: {example_path}")
    
    print(f"\n To use these hyperparameters in another script:")
    print(f"  1. Copy 'optuna_results/best_hyperparameters.json' to your project")
    print(f"  2. Use the example in 'optuna_results/use_optimized_hyperparameters.py'")
    print(f"  3. Or manually copy the hyperparameters shown above")
    
    print(f"\n  Note: These hyperparameters were optimized with 3.5 minute timeout per trial.")
    print(f"   For final training, you may want to increase epochs to 30-50.")
    
    return hyperparameters


if __name__ == "__main__":
    hyperparameters = main()
    
    if hyperparameters:
        print(f"\n Best hyperparameters found!")
        print(f"   ROC AUC: {hyperparameters['best_roc_auc']:.4f}")
        print(f"   Parameters: {hyperparameters['total_parameters']:,}")
        print(f"\nReady to train with these hyperparameters in another script! ")