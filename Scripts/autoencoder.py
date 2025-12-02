"""
Autoencoder for Leukemia Detection (Anomaly Detection Approach)
Input: 450x450x1 Grayscale images
Training: Only on Healthy (Hem) cells
Inference: Detects Leukemia (All) cells as anomalies (high reconstruction error)
Includes: Logging for Loss and Time (No Accuracy during training)
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

# ============ LOGGING UTILS (Modified: No Accuracy) ============

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
        f.write("REPORTE DE RENDIMIENTO - AUTOENCODER (LOSS & TIME ONLY)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Timestamp: {metrics['timestamp']}\n")
        
        f.write("-" * 80 + "\nANALISIS COMPUTACIONAL\n" + "-" * 80 + "\n")
        f.write(f"Total parametros: {metrics['parametros']['total_parametros']:,}\n")
        f.write(f"Throughput inferencia: {metrics['tiempo_inferencia']['throughput_imagenes_por_segundo']:.2f} img/s\n\n")
        
        f.write("-" * 80 + "\nRENDIMIENTO FINAL\n" + "-" * 80 + "\n")
        f.write(f"Final Reconstruction Loss (MSE): {metrics['rendimiento_entrenamiento']['final_train_loss']:.6f}\n")
        f.write(f"Total Training Time: {metrics['rendimiento_entrenamiento']['total_training_time_seconds']:.2f} s\n")
    
    print(f"Reporte guardado en: {filepath}")

def generar_graficas_entrenamiento(metrics, output_path, timestamp):
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Reconstruction Loss
    plt.subplot(1, 2, 1)
    plt.plot(metrics['metricas_por_epoch']['train_losses'], label='Train Loss (MSE)', color='blue', linewidth=2)
    plt.title('Reconstruction Loss Evolution')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Time per Epoch
    plt.subplot(1, 2, 2)
    plt.plot(metrics['metricas_por_epoch']['epoch_times'], color='purple', linewidth=2)
    plt.title('Time per Epoch (seconds)')
    plt.xlabel('Epoch')
    plt.ylabel('Seconds')
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
            'total_training_time_seconds': sum(metrics['epoch_times'])
        },
        'metricas_por_epoch': metrics
    }
    
    generar_reporte_completo(all_metrics, output_path, timestamp)
    generar_graficas_entrenamiento(all_metrics, output_path, timestamp)
    return all_metrics

# ============ TRAINING LOOP ============

def train_autoencoder(model, train_loader, epochs=30, lr=0.001, device='cuda'):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Store metrics for logging
    metrics = {'train_losses': [], 'epoch_times': []}
    
    print(f"\nIniciando entrenamiento de Autoencoder en {device}...")
    print(f"{'Epoch':^6} | {'Loss (MSE)':^12} | {'Tiempo (s)':^10}")
    print("-" * 35)
    
    for epoch in range(epochs):
        start_time = time.time()
        
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
            
        avg_loss = train_loss / len(train_loader)
        
        end_time = time.time()
        epoch_time = end_time - start_time
        
        # Save metrics
        metrics['train_losses'].append(avg_loss)
        metrics['epoch_times'].append(epoch_time)
        
        print(f"{epoch+1:^6} | {avg_loss:^12.6f} | {epoch_time:^10.2f}")
            
    return model, metrics

# ============ MAIN ============

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Data
    print("Cargando datasets...")
    try:
        datasets, _ = cargar_todos_datasets_con_labels()
    except:
        print("Using fallback loading...")
        datasets = {
            'fold_0_hem': cargar_training_hem_original("data/training_data/fold_0/hem/", max_imagenes=200),
            'fold_0_all': cargar_training_all_original("data/training_data/fold_0/all/", max_imagenes=200)
        }

    # 2. Prepare Data
    # Note: We still need test_loader for the FINAL evaluation, but not for the training loop
    train_loader, test_loader, _ = prepare_anomaly_data(datasets, batch_size=16)
    
    # 3. Create Model
    model = LeukemiaAutoencoder()
    
    # 4. Train (NOW WITH LOGS but WITHOUT ACCURACY)
    model, metrics = train_autoencoder(model, train_loader, epochs=30, device=device)
    
    # 5. Generate Logs
    print(f"\nGenerando logs y reportes de entrenamiento...")
    analizar_rendimiento_computacional(model, metrics, "logs_autoencoder")

    # 6. Final Evaluation (Only done once at the end)
    # Here we DO check classification performance, but it doesn't affect the logs created above
    
    # Helper functions for final evaluation (local scope)
    def evaluate_final(model, test_loader, device):
        model.eval()
        criterion = nn.MSELoss(reduction='none')
        errors, labels = [], []
        with torch.no_grad():
            for data, label in test_loader:
                data = data.to(device)
                recon = model(data)
                loss = criterion(recon, data).mean(dim=(1,2,3))
                errors.extend(loss.cpu().numpy())
                labels.extend(label.numpy())
        return np.array(errors), np.array(labels)

    errors, true_labels = evaluate_final(model, test_loader, device)
    
    # Find Threshold
    fpr, tpr, thresholds = roc_curve(true_labels, errors)
    optimal_idx = np.argmax(tpr - fpr)
    threshold = thresholds[optimal_idx]
    roc_auc = auc(fpr, tpr)
    predicted_labels = (errors > threshold).astype(int)
    
    print("\n" + "="*50)
    print("RESULTADOS FINAL DETECCION DE ANOMALIAS")
    print("="*50)
    print(f"Optimal Threshold (MSE): {threshold:.6f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_labels, target_names=['Healthy', 'Leukemia']))
    
    # Save model
    torch.save(model.state_dict(), 'leukemia_autoencoder.pth')
    print("Modelo guardado como leukemia_autoencoder.pth")

if __name__ == "__main__":
    main()