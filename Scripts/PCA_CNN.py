"""
CNN Model for Leukemia Classification - Combined Training Data PCA
Input: 450x450x1 Grayscale images from PCA (Healthy vs Leukemia cells)
Output: Binary classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import cv2

# Importar funciones de carga de datos
from Creador_labels import cargar_todos_datasets_con_labels_PCA


class LeukemiaCNN(nn.Module):
    """
    Recommended CNN Architecture for 450x450 grayscale images from PCA
    
    Architecture reasoning:
    - Input: 450x450x1 → We have high-resolution medical images in grayscale from PCA reconstruction
    - Strategy: Progressive downsampling with increasing channels  
    - Goal: Extract cell morphology features for classification
    - NOTE: Data converted to grayscale from PCA RGB data
    """
    
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super(LeukemiaCNN, self).__init__()
        
        # ============ Feature Extraction Blocks ============
        
        # Block 1: 450x450 → 225x225
        # Purpose: Extract low-level features (edges, textures)
        # NOTE: Input channels = 1 for grayscale PCA data
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),      # 450x450x32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),     # 450x450x32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),           # 225x225x32
        )
        
        # Block 2: 225x225 → 112x112
        # Purpose: Extract intermediate features (cell components)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),     # 225x225x64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),     # 225x225x64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),           # 112x112x64
        )
        
        # Block 3: 112x112 → 56x56
        # Purpose: Extract higher-level features (cell patterns)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),    # 112x112x128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),   # 112x112x128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),           # 56x56x128
        )
        
        # Block 4: 56x56 → 28x28
        # Purpose: Extract abstract features
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),   # 56x56x256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),   # 56x56x256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),           # 28x28x256
        )
        
        # ============ Global Average Pooling ============
        # Reduces 28x28x256 → 1x1x256
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # ============ Classification Head ============
        # Purpose: Map extracted features to class probabilities
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, num_classes)  # Output: 2 classes (Healthy/Leukemia)
        )
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, 1, 450, 450)  # Grayscale from PCA
        
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Feature extraction
        x = self.block1(x)  # 450x450 → 225x225
        x = self.block2(x)  # 225x225 → 112x112
        x = self.block3(x)  # 112x112 → 56x56
        x = self.block4(x)  # 56x56 → 28x28
        
        # Global pooling
        x = self.avgpool(x)  # 28x28x256 → 1x1x256
        
        # Flatten for classification
        x = torch.flatten(x, 1)  # 1x1x256 → 256
        
        # Classification
        x = self.classifier(x)  # 256 → num_classes
        
        return x


# ============ DATA PREPARATION FUNCTIONS ============

def convertir_a_escala_grises(imagenes):
    """
    Convert RGB images to grayscale
    """
    imagenes_gris = []
    for img in imagenes:
        if len(img.shape) == 3 and img.shape[2] == 3:
            # Convert RGB to grayscale
            img_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            # Already grayscale or single channel
            img_gris = img if len(img.shape) == 2 else img[:, :, 0]
        
        # Add channel dimension
        img_gris = np.expand_dims(img_gris, axis=-1)
        imagenes_gris.append(img_gris)
    
    return imagenes_gris


def preparar_dataloaders_todos_los_folds(datasets, labels, batch_size=32, val_size=0.2):
    """
    Prepare DataLoaders combining ALL folds into single training dataset
    NOTE: PCA data converted from RGB to grayscale
    """
    # Combine images and labels from ALL folds
    all_images = []
    all_labels = []
    
    for fold in ['fold_0', 'fold_1', 'fold_2']:
        all_images.extend(datasets[f'{fold}_all'])
        all_images.extend(datasets[f'{fold}_hem'])
        all_labels.extend(labels[f'{fold}_all'])
        all_labels.extend(labels[f'{fold}_hem'])
    
    # Convert to grayscale (PCA data comes as RGB, need to convert to 1 channel)
    print("  - Convirtiendo imagenes PCA a escala de grises...")
    all_images_gray = convertir_a_escala_grises(all_images)
    
    # Convert to numpy arrays
    X = np.array(all_images_gray)
    y = np.array(all_labels)
    
    print(f"\nPreparando DataLoaders con TODOS los folds combinados (PCA):")
    print(f"  - Total imagenes: {len(X)}")
    print(f"  - Shape de imagenes: {X[0].shape}")  # Should be (450, 450, 1) - Grayscale from PCA
    print(f"  - Healthy (0): {np.sum(y == 0)}")
    print(f"  - Leukemia (1): {np.sum(y == 1)}")
    
    # Split into train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, random_state=42, stratify=y
    )
    
    # Convert to PyTorch tensors and normalize
    X_train = torch.FloatTensor(X_train).permute(0, 3, 1, 2) / 255.0  # (N, H, W, 1) → (N, 1, H, W)
    X_val = torch.FloatTensor(X_val).permute(0, 3, 1, 2) / 255.0
    y_train = torch.LongTensor(y_train)
    y_val = torch.LongTensor(y_val)
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"  - Entrenamiento: {len(X_train)} imagenes")
    print(f"  - Validacion: {len(X_val)} imagenes")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Formato de entrada: {X_train.shape}")  # Should be (N, 1, 450, 450)
    
    return train_loader, val_loader, (X_train, y_train, X_val, y_val)


# ============ TRAINING FUNCTIONS ============

def entrenar_modelo_con_metricas(model, train_loader, val_loader, epochs=25, lr=0.001, device='cuda'):
    """
    Train model with comprehensive metrics tracking for 25 epochs
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Metrics storage
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    epoch_times = []
    
    print(f"\nIniciando entrenamiento en {device}...")
    print(f"{'Epoch':^6} | {'Train Loss':^10} | {'Train Acc':^10} | {'Val Loss':^10} | {'Val Acc':^10} | {'Time (s)':^10}")
    print("-" * 75)
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        # Calculate metrics
        train_loss_avg = train_loss / len(train_loader)
        val_loss_avg = val_loss / len(val_loader)
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        epoch_time = time.time() - epoch_start_time
        
        # Store metrics
        train_losses.append(train_loss_avg)
        val_losses.append(val_loss_avg)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        epoch_times.append(epoch_time)
        
        # Update scheduler
        scheduler.step()
        
        # Print progress
        print(f"{epoch+1:^6} | {train_loss_avg:^10.4f} | {train_acc:^10.2f} | {val_loss_avg:^10.4f} | {val_acc:^10.2f} | {epoch_time:^10.2f}")
    
    metrics = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'epoch_times': epoch_times,
        'final_val_acc': val_acc,
        'final_val_loss': val_loss_avg
    }
    
    return model, metrics


# ============ COMPUTATIONAL ANALYSIS FUNCTIONS ============

def contar_parametros_modelo(model):
    """
    Cuenta el numero total de parametros del modelo
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total_params - trainable_params
    
    model_size_mb = (total_params * 4) / (1024 ** 2)
    
    return {
        'total_parametros': total_params,
        'parametros_entrenables': trainable_params,
        'parametros_no_entrenables': non_trainable,
        'tamaño_modelo_mb': model_size_mb
    }


def medir_tiempo_inferencia(model, batch_size=32, num_iterations=50):
    """
    Mide el tiempo de inferencia del modelo
    """
    device = next(model.parameters()).device
    model.eval()
    
    # Input shape for grayscale (1 channel for PCA data)
    dummy_input = torch.randn(batch_size, 1, 450, 450).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Medicion
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time_ms = (total_time / num_iterations) * 1000
    throughput = (num_iterations * batch_size) / total_time
    
    return {
        'batch_size': batch_size,
        'num_iteraciones': num_iterations,
        'tiempo_total_segundos': total_time,
        'tiempo_promedio_ms': avg_time_ms,
        'throughput_imagenes_por_segundo': throughput,
        'dispositivo': str(device)
    }


def analizar_rendimiento_computacional(model, metrics, output_path="logs_pca_combined"):
    """
    Comprehensive computational and performance analysis for PCA data
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Computational metrics
    parametros = contar_parametros_modelo(model)
    tiempo_inferencia = medir_tiempo_inferencia(model)
    
    # Performance metrics from training
    final_train_acc = metrics['train_accs'][-1]
    final_val_acc = metrics['val_accs'][-1]
    final_train_loss = metrics['train_losses'][-1]
    final_val_loss = metrics['val_losses'][-1]
    avg_epoch_time = np.mean(metrics['epoch_times'])
    
    # Compile all metrics
    all_metrics = {
        'timestamp': datetime.now().isoformat(),
        'parametros': parametros,
        'tiempo_inferencia': tiempo_inferencia,
        'rendimiento_entrenamiento': {
            'final_train_accuracy': final_train_acc,
            'final_val_accuracy': final_val_acc,
            'final_train_loss': final_train_loss,
            'final_val_loss': final_val_loss,
            'avg_epoch_time_seconds': avg_epoch_time,
            'total_training_time_seconds': sum(metrics['epoch_times'])
        },
        'metricas_por_epoch': {
            'train_losses': metrics['train_losses'],
            'val_losses': metrics['val_losses'],
            'train_accuracies': metrics['train_accs'],
            'val_accuracies': metrics['val_accs'],
            'epoch_times': metrics['epoch_times']
        },
        'configuracion': 'PCA_GRAYSCALE_450x450x1_COMBINED'
    }
    
    # Generate comprehensive report
    generar_reporte_completo(all_metrics, output_path, timestamp)
    
    # Generate visualization
    generar_graficas_entrenamiento(metrics, output_path, timestamp)
    
    return all_metrics


def generar_reporte_completo(metrics, output_path, timestamp):
    """
    Generate comprehensive performance report for PCA data
    """
    filename = f"reporte_rendimiento_pca_combined_{timestamp}.txt"
    filepath = os.path.join(output_path, filename)
    
    with open(filepath, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("REPORTE COMPLETO DE RENDIMIENTO - CNN LEUKEMIA CLASSIFICATION (PCA DATA)\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Timestamp: {metrics['timestamp']}\n")
        f.write(f"Modelo: LeukemiaCNN - PCA Grayscale\n")
        f.write(f"Configuracion: {metrics['configuracion']}\n\n")
        
        # Computational Analysis
        f.write("-" * 80 + "\n")
        f.write("ANALISIS COMPUTACIONAL\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total parametros: {metrics['parametros']['total_parametros']:,}\n")
        f.write(f"Parametros entrenables: {metrics['parametros']['parametros_entrenables']:,}\n")
        f.write(f"Tamaño del modelo: {metrics['parametros']['tamaño_modelo_mb']:.2f} MB\n")
        f.write(f"Throughput inferencia: {metrics['tiempo_inferencia']['throughput_imagenes_por_segundo']:.2f} img/s\n")
        f.write(f"Latencia inferencia: {metrics['tiempo_inferencia']['tiempo_promedio_ms']:.2f} ms\n\n")
        
        # Training Performance
        f.write("-" * 80 + "\n")
        f.write("RENDIMIENTO DE ENTRENAMIENTO\n")
        f.write("-" * 80 + "\n")
        f.write(f"Accuracy final entrenamiento: {metrics['rendimiento_entrenamiento']['final_train_accuracy']:.2f}%\n")
        f.write(f"Accuracy final validacion: {metrics['rendimiento_entrenamiento']['final_val_accuracy']:.2f}%\n")
        f.write(f"Loss final entrenamiento: {metrics['rendimiento_entrenamiento']['final_train_loss']:.4f}\n")
        f.write(f"Loss final validacion: {metrics['rendimiento_entrenamiento']['final_val_loss']:.4f}\n")
        f.write(f"Tiempo promedio por epoca: {metrics['rendimiento_entrenamiento']['avg_epoch_time_seconds']:.2f} s\n")
        f.write(f"Tiempo total entrenamiento: {metrics['rendimiento_entrenamiento']['total_training_time_seconds']:.2f} s\n\n")
        
        # Performance Analysis
        f.write("-" * 80 + "\n")
        f.write("ANALISIS DE RENDIMIENTO\n")
        f.write("-" * 80 + "\n")
        
        train_acc = metrics['rendimiento_entrenamiento']['final_train_accuracy']
        val_acc = metrics['rendimiento_entrenamiento']['final_val_accuracy']
        train_loss = metrics['rendimiento_entrenamiento']['final_train_loss']
        val_loss = metrics['rendimiento_entrenamiento']['final_val_loss']
        
        # Overfitting analysis
        accuracy_gap = train_acc - val_acc
        loss_gap = val_loss - train_loss
        
        f.write(f"Brecha de accuracy (train-val): {accuracy_gap:.2f}%\n")
        f.write(f"Brecha de loss (val-train): {loss_gap:.4f}\n")
        
        if accuracy_gap > 10:
            f.write("ALTO OVERFITTING: Brecha de accuracy > 10%\n")
        elif accuracy_gap > 5:
            f.write("OVERFITTING MODERADO: Brecha de accuracy > 5%\n")
        else:
            f.write("BUEN AJUSTE: Brecha de accuracy aceptable\n")
            
        if val_acc >= 90:
            f.write("EXCELENTE RENDIMIENTO: Accuracy > 90%\n")
        elif val_acc >= 80:
            f.write("BUEN RENDIMIENTO: Accuracy > 80%\n")
        else:
            f.write("NECESITA MEJORAS: Accuracy < 80%\n")
    
    print(f"Reporte completo guardado en: {filepath}")


def generar_graficas_entrenamiento(metrics, output_path, timestamp):
    """
    Generate training visualization graphs for PCA data
    """
    plt.figure(figsize=(15, 10))
    
    # Loss plot
    plt.subplot(2, 2, 1)
    plt.plot(metrics['train_losses'], label='Train Loss', linewidth=2)
    plt.plot(metrics['val_losses'], label='Val Loss', linewidth=2)
    plt.title('Evolucion de la Perdida - PCA Data', fontsize=14, fontweight='bold')
    plt.xlabel('Epoca')
    plt.ylabel('Perdida')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Accuracy plot
    plt.subplot(2, 2, 2)
    plt.plot(metrics['train_accs'], label='Train Accuracy', linewidth=2)
    plt.plot(metrics['val_accs'], label='Val Accuracy', linewidth=2)
    plt.title('Evolucion del Accuracy - PCA Data', fontsize=14, fontweight='bold')
    plt.xlabel('Epoca')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Epoch time plot
    plt.subplot(2, 2, 3)
    plt.plot(metrics['epoch_times'], color='purple', linewidth=2)
    plt.title('Tiempo por Epoca - PCA Data', fontsize=14, fontweight='bold')
    plt.xlabel('Epoca')
    plt.ylabel('Tiempo (segundos)')
    plt.grid(True, alpha=0.3)
    
    # Combined performance plot
    plt.subplot(2, 2, 4)
    epochs = range(1, len(metrics['train_losses']) + 1)
    plt.plot(epochs, metrics['train_losses'], 'b-', label='Train Loss', alpha=0.7)
    plt.plot(epochs, metrics['val_losses'], 'r-', label='Val Loss', alpha=0.7)
    plt.twinx()
    plt.plot(epochs, metrics['train_accs'], 'g--', label='Train Acc', alpha=0.7)
    plt.plot(epochs, metrics['val_accs'], 'm--', label='Val Acc', alpha=0.7)
    plt.title('Metricas Combinadas - PCA Data', fontsize=14, fontweight='bold')
    plt.xlabel('Epoca')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'graficas_entrenamiento_pca_{timestamp}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Graficas de entrenamiento PCA guardadas")


# ============ MAIN EXECUTION ============

def main():
    """
    Main execution function - Combined training with PCA data
    """
    print("INICIANDO ENTRENAMIENTO CNN CON DATOS PCA")
    print(f"{'='*60}")
    
    # Hyperparameter configuration
    hyperparams = {
        'lr': 0.001,
        'dropout_rate': 0.5,
        'batch_size': 32,
        'epochs': 25
    }
    
    print("\nCONFIGURACION DE HIPERPARAMETROS:")
    print(f"  - Learning rate: {hyperparams['lr']}")
    print(f"  - Dropout rate: {hyperparams['dropout_rate']}")
    print(f"  - Batch size: {hyperparams['batch_size']}")
    print(f"  - Epochs: {hyperparams['epochs']}")
    
    # Load PCA data
    print(f"\nCARGANDO DATOS PCA DE TODOS LOS FOLDS...")
    datasets, labels = cargar_todos_datasets_con_labels_PCA()
    
    # Prepare data loaders with ALL folds combined
    print(f"\nCOMBINANDO TODOS LOS FOLDS PCA EN UN SOLO DATASET...")
    train_loader, val_loader, _ = preparar_dataloaders_todos_los_folds(
        datasets, labels, batch_size=hyperparams['batch_size']
    )
    
    # Create and train model
    print(f"\nCREANDO Y ENTRENANDO MODELO CON DATOS PCA...")
    model = LeukemiaCNN(
        num_classes=2, 
        dropout_rate=hyperparams['dropout_rate']
    )
    
    model_trained, metrics = entrenar_modelo_con_metricas(
        model, train_loader, val_loader, 
        epochs=hyperparams['epochs'], 
        lr=hyperparams['lr']
    )
    
    # Comprehensive analysis
    print(f"\nANALISIS COMPLETO DE RENDIMIENTO PCA")
    all_metrics = analizar_rendimiento_computacional(model_trained, metrics, "logs_pca_combined_training")
    
    print(f"\nENTRENAMIENTO Y ANALISIS COMPLETADOS EXITOSAMENTE!")
    print(f"Configuracion: PCA Grayscale 450x450x1 - Datos Combinados")
    print(f"Epochs: {hyperparams['epochs']}")
    print(f"Accuracy final validacion: {metrics['val_accs'][-1]:.2f}%")
    print(f"Loss final validacion: {metrics['val_losses'][-1]:.4f}")
    print(f"Reportes guardados en: logs_pca_combined_training/")


if __name__ == "__main__":
    main()