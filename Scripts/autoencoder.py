"""
Autoencoder for Leukemia Detection (Anomaly Detection Approach)
Input: 450x450x1 Grayscale images
Training: Only on Healthy (Hem) cells
Inference: Detects Leukemia (All) cells as anomalies (high reconstruction error)
Includes: Full logging and Accuracy metrics per epoch
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
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
import cv2

# Import data loading functions
try:
    from Creador_labels import cargar_todos_datasets_con_labels
except ImportError:
    print("Warning: Creador_labels not found. Ensure data loading functions are available.")

from Carga_imagenes import cargar_training_all_original, cargar_training_hem_original


class LeukemiaAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for 450x450 Grayscale Images.
    """
    def __init__(self):
        super(LeukemiaAutoencoder, self).__init__()
        
        # ============ ENCODER ============
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2), 
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2) 
        )
        
        # ============ DECODER ============
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.Upsample(size=(225, 225), mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.Upsample(size=(450, 450), mode='bilinear', align_corners=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1), 
            nn.Sigmoid() 
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# ============ DATA PREPARATION ============

def convert_to_grayscale(images):
    processed = []
    for img in images:
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        processed.append(np.expand_dims(gray, axis=-1))
    return np.array(processed)

def prepare_anomaly_data(datasets, batch_size=16):
    print("Preparando datos para Deteccion de Anomalias...")
    
    hem_images = []
    for key in datasets:
        if 'hem' in key:
            hem_images.extend(datasets[key])
            
    all_images = []
    for key in datasets:
        if 'all' in key:
            all_images.extend(datasets[key])
            
    hem_gray = convert_to_grayscale(hem_images)
    all_gray = convert_to_grayscale(all_images)
    
    hem_gray = hem_gray.astype('float32') / 255.0
    all_gray = all_gray.astype('float32') / 255.0
    
    # 80% Train (Healthy only), 20% Test (Healthy)
    X_train_healthy, X_test_healthy = train_test_split(hem_gray, test_size=0.2, random_state=42)
    
    # Test set = Reserved Healthy + All Leukemia
    X_test = np.concatenate([X_test_healthy, all_gray], axis=0)
    y_test = np.concatenate([np.zeros(len(X_test_healthy)), np.ones(len(all_gray))], axis=0)
    
    X_train_tensor = torch.FloatTensor(X_train_healthy).permute(0, 3, 1, 2)
    X_test_tensor = torch.FloatTensor(X_test).permute(0, 3, 1, 2)
    y_test_tensor = torch.LongTensor(y_test)
    
    train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"  - Datos Entrenamiento (Solo Healthy): {len(X_train_tensor)} imagenes")
    print(f"  - Datos Test (Mixto): {len(X_test_tensor)} imagenes")
    
    return train_loader, test_loader, (X_train_tensor, X_test_tensor, y_test_tensor)

# ============ METRICS & LOGGING (Adapted from CNN.py) ============

def contar_parametros_modelo(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = (total_params * 4) / (1024 ** 2)
    return {
        'total_parametros': total_params,
        'parametros_entrenables': trainable_params,
        'tamaño_modelo_mb': model_size_mb
    }

def medir_tiempo_inferencia(model, batch_size=16, num_iterations=50):
    device = next(model.parameters()).device
    model.eval()
    dummy_input = torch.randn(batch_size, 1, 450, 450).to(device)
    
    with torch.no_grad():
        for _ in range(10): _ = model(dummy_input) # Warmup
    
    if device.type == 'cuda': torch.cuda.synchronize()
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations): _ = model(dummy_input)
    if device.type == 'cuda': torch.cuda.synchronize()
    
    total_time = time.time() - start_time
    return {
        'tiempo_promedio_ms': (total_time / num_iterations) * 1000,
        'throughput_imagenes_por_segundo': (num_iterations * batch_size) / total_time
    }

def generar_reporte_completo(metrics, output_path, timestamp):
    filename = f"reporte_rendimiento_autoencoder_{timestamp}.txt"
    filepath = os.path.join(output_path, filename)
    
    with open(filepath, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("REPORTE DE RENDIMIENTO - AUTOENCODER ANOMALY DETECTION\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Timestamp: {metrics['timestamp']}\n")
        
        f.write("-" * 80 + "\nANALISIS COMPUTACIONAL\n" + "-" * 80 + "\n")
        f.write(f"Total parametros: {metrics['parametros']['total_parametros']:,}\n")
        f.write(f"Throughput inferencia: {metrics['tiempo_inferencia']['throughput_imagenes_por_segundo']:.2f} img/s\n\n")
        
        f.write("-" * 80 + "\nRENDIMIENTO FINAL\n" + "-" * 80 + "\n")
        f.write(f"Best Validation Accuracy: {max(metrics['metricas_por_epoch']['val_accuracies']):.2f}%\n")
        f.write(f"Final Validation Loss (Reconstruction MSE): {metrics['rendimiento_entrenamiento']['final_val_loss']:.6f}\n")
        f.write(f"Total Training Time: {metrics['rendimiento_entrenamiento']['total_training_time_seconds']:.2f} s\n")
    
    print(f"Reporte guardado en: {filepath}")

def generar_graficas_entrenamiento(metrics, output_path, timestamp):
    plt.figure(figsize=(15, 10))
    
    # Loss
    plt.subplot(2, 2, 1)
    plt.plot(metrics['metricas_por_epoch']['train_losses'], label='Train Loss (Healthy)', color='blue')
    plt.plot(metrics['metricas_por_epoch']['val_losses'], label='Val Loss (Mixed)', color='orange')
    plt.title('Reconstruction Loss (MSE)')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Accuracy (Anomaly Detection Performance)
    plt.subplot(2, 2, 2)
    plt.plot(metrics['metricas_por_epoch']['val_accuracies'], label='Val Accuracy (Detection)', color='green')
    plt.title('Anomaly Detection Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Epoch Times
    plt.subplot(2, 2, 3)
    plt.plot(metrics['metricas_por_epoch']['epoch_times'], color='purple')
    plt.title('Time per Epoch (seconds)')
    plt.xlabel('Epoch')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'graficas_autoencoder_{timestamp}.png'), dpi=300)
    plt.close()
    print("Graficas guardadas.")

def analizar_rendimiento_computacional(model, metrics, output_path="logs_autoencoder"):
    if not os.path.exists(output_path): os.makedirs(output_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    all_metrics = {
        'timestamp': datetime.now().isoformat(),
        'parametros': contar_parametros_modelo(model),
        'tiempo_inferencia': medir_tiempo_inferencia(model),
        'rendimiento_entrenamiento': {
            'final_train_loss': metrics['train_losses'][-1],
            'final_val_loss': metrics['val_losses'][-1],
            'total_training_time_seconds': sum(metrics['epoch_times'])
        },
        'metricas_por_epoch': metrics
    }
    
    generar_reporte_completo(all_metrics, output_path, timestamp)
    generar_graficas_entrenamiento(all_metrics, output_path, timestamp)
    return all_metrics

# ============ TRAINING & EVALUATION LOOP ============

def evaluate_and_get_accuracy(model, test_loader, device):
    """Calculates reconstruction error and determines best accuracy for this epoch"""
    model.eval()
    criterion = nn.MSELoss(reduction='none')
    
    errors = []
    labels = []
    val_loss_sum = 0
    
    with torch.no_grad():
        for data, label in test_loader:
            data = data.to(device)
            reconstruction = model(data)
            
            # Loss for tracking
            batch_loss = nn.MSELoss()(reconstruction, data).item()
            val_loss_sum += batch_loss
            
            # Error per image for classification
            loss_per_img = criterion(reconstruction, data).mean(dim=(1,2,3))
            errors.extend(loss_per_img.cpu().numpy())
            labels.extend(label.numpy())
            
    avg_val_loss = val_loss_sum / len(test_loader)
    
    # Calculate Accuracy based on optimal threshold
    fpr, tpr, thresholds = roc_curve(labels, errors)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    preds = (np.array(errors) > optimal_threshold).astype(int)
    accuracy = (preds == np.array(labels)).mean() * 100
    
    return avg_val_loss, accuracy

def train_autoencoder(model, train_loader, test_loader, epochs=30, lr=0.001, device='cuda'):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Metrics containers
    metrics = {
        'train_losses': [], 'val_losses': [],
        'val_accuracies': [], 'epoch_times': []
    }
    
    print(f"\nIniciando entrenamiento de Autoencoder en {device}...")
    print(f"{'Epoch':^6} | {'Train Loss':^12} | {'Val Loss':^12} | {'Val Acc %':^10} | {'Time (s)':^10}")
    print("-" * 65)
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # 1. Training Phase
        model.train()
        train_loss = 0.0
        for data, _ in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            reconstruction = model(data)
            loss = criterion(reconstruction, data)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # 2. Validation Phase (Calculate Accuracy)
        avg_val_loss, val_acc = evaluate_and_get_accuracy(model, test_loader, device)
        
        end_time = time.time()
        epoch_time = end_time - start_time
        
        # Store metrics
        metrics['train_losses'].append(avg_train_loss)
        metrics['val_losses'].append(avg_val_loss)
        metrics['val_accuracies'].append(val_acc)
        metrics['epoch_times'].append(epoch_time)
        
        print(f"{epoch+1:^6} | {avg_train_loss:^12.6f} | {avg_val_loss:^12.6f} | {val_acc:^10.2f} | {epoch_time:^10.2f}")
            
    return model, metrics

# ============ MAIN ============

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Cargando datasets...")
    # Load data (Assuming script is in Scripts/ folder)
    try:
        datasets, _ = cargar_todos_datasets_con_labels()
    except:
        print("Using fallback loading...")
        datasets = {
            'fold_0_hem': cargar_training_hem_original("../data/training_data/fold_0/hem/", max_imagenes=200),
            'fold_0_all': cargar_training_all_original("../data/training_data/fold_0/all/", max_imagenes=200)
        }

    # Prepare data
    train_loader, test_loader, _ = prepare_anomaly_data(datasets, batch_size=16)
    
    # Create Model
    model = LeukemiaAutoencoder()
    
    # Train with metrics
    model, metrics = train_autoencoder(model, train_loader, test_loader, epochs=30, device=device)
    
    # Generate Logs and Analysis
    print(f"\nGenerando logs y reportes como en CNN.py...")
    analizar_rendimiento_computacional(model, metrics, "logs_autoencoder")
    
    # Save model
    torch.save(model.state_dict(), 'leukemia_autoencoder_final.pth')
    print("Modelo guardado exitosamente.")

if __name__ == "__main__":
    main()