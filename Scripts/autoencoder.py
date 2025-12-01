"""
Autoencoder for Leukemia Detection (Anomaly Detection Approach)
Input: 450x450x1 Grayscale images
Training: Only on Healthy (Hem) cells
Inference: Detects Leukemia (All) cells as anomalies (high reconstruction error)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split
import cv2

# Import data loading functions
# Assuming Creador_labels loads the dictionary separated by folds and types
try:
    from Creador_labels import cargar_todos_datasets_con_labels
except ImportError:
    # Fallback if Creador_labels is not available
    print("Warning: Creador_labels not found. Ensure data loading functions are available.")

from Carga_imagenes import cargar_training_all_original, cargar_training_hem_original


class LeukemiaAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for 450x450 Grayscale Images.
    Uses nn.Upsample to handle odd dimensions (450 -> 225) correctly during reconstruction.
    """
    def __init__(self):
        super(LeukemiaAutoencoder, self).__init__()
        
        # ============ ENCODER ============
        # Compresses the image into a latent representation
        self.encoder = nn.Sequential(
            # Block 1: 450x450 -> 225x225
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2), 
            
            # Block 2: 225x225 -> 112x112
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            
            # Block 3: 112x112 -> 56x56
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            
            # Block 4: 56x56 -> 28x28 (Latent Space)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2) 
        )
        
        # ============ DECODER ============
        # Reconstructs the image from the latent representation
        self.decoder = nn.Sequential(
            # Block 4 Up: 28x28 -> 56x56
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # Block 3 Up: 56x56 -> 112x112
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # Block 2 Up: 112x112 -> 225x225
            # Note: 112*2 = 224, so we need to force size to 225 if matching original CNN logic
            # However, standard upsampling gives 224. We will rely on adaptive sizing or padding if strict 450 is needed.
            # Here we use explicit size in Upsample to ensure we get back to 450x450 eventually.
            nn.Upsample(size=(225, 225), mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            # Block 1 Up: 225x225 -> 450x450
            nn.Upsample(size=(450, 450), mode='bilinear', align_corners=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1), # Output 1 channel (Grayscale)
            nn.Sigmoid() # Normalize output to [0, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# ============ DATA PREPARATION ============

def convert_to_grayscale(images):
    """Helper to convert list of BGR images to grayscale (N, 450, 450, 1)"""
    processed = []
    for img in images:
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        processed.append(np.expand_dims(gray, axis=-1))
    return np.array(processed)

def prepare_anomaly_data(datasets, batch_size=16):
    """
    Prepares data specifically for Anomaly Detection.
    
    1. Train Set: ONLY Healthy (Hem) images.
    2. Test Set: Mixture of Healthy (Hem) and Leukemia (All).
    """
    print("Preparando datos para Deteccion de Anomalias...")
    
    # 1. Extract ALL Healthy images (Training Data)
    hem_images = []
    for key in datasets:
        if 'hem' in key: # Healthy
            hem_images.extend(datasets[key])
            
    # 2. Extract ALL Leukemia images (Anomalies for Testing)
    all_images = []
    for key in datasets:
        if 'all' in key: # Leukemia
            all_images.extend(datasets[key])
            
    # Convert to Grayscale
    hem_gray = convert_to_grayscale(hem_images)
    all_gray = convert_to_grayscale(all_images)
    
    # Normalize to [0, 1]
    hem_gray = hem_gray.astype('float32') / 255.0
    all_gray = all_gray.astype('float32') / 255.0
    
    # Split Healthy data: 
    # 80% for Training (Learning "Normal"), 20% for Testing (Checking False Positives)
    X_train_healthy, X_test_healthy = train_test_split(hem_gray, test_size=0.2, random_state=42)
    
    # The Test set contains the reserved Healthy images (Label 0) AND All Leukemia images (Label 1)
    X_test = np.concatenate([X_test_healthy, all_gray], axis=0)
    
    # Labels for Testing (0 = Normal/Healthy, 1 = Anomaly/Leukemia)
    y_test = np.concatenate([np.zeros(len(X_test_healthy)), np.ones(len(all_gray))], axis=0)
    
    # Create PyTorch Tensors
    # Permute to (N, Channels, Height, Width)
    X_train_tensor = torch.FloatTensor(X_train_healthy).permute(0, 3, 1, 2)
    X_test_tensor = torch.FloatTensor(X_test).permute(0, 3, 1, 2)
    y_test_tensor = torch.LongTensor(y_test)
    
    # DataLoaders
    # For training, we only need X vs X (reconstruction), so target is X
    train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor) # X vs Label (for evaluation)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"  - Datos Entrenamiento (Solo Healthy): {len(X_train_tensor)} imagenes")
    print(f"  - Datos Test (Mixto): {len(X_test_tensor)} imagenes")
    print(f"    - Healthy (Test): {len(X_test_healthy)}")
    print(f"    - Leukemia (Anomalies): {len(all_gray)}")
    
    return train_loader, test_loader

# ============ TRAINING ============

def train_autoencoder(model, train_loader, epochs=50, lr=0.001, device='cuda'):
    model = model.to(device)
    criterion = nn.MSELoss() # Reconstruction Error
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {'loss': []}
    
    print(f"\nIniciando entrenamiento de Autoencoder en {device}...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for data, _ in train_loader: # Target is the input itself
            data = data.to(device)
            
            optimizer.zero_grad()
            reconstruction = model(data)
            loss = criterion(reconstruction, data)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_loss = train_loss / len(train_loader)
        history['loss'].append(avg_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Reconstruction Loss: {avg_loss:.6f}")
            
    return model, history

# ============ EVALUATION ============

def evaluate_anomaly_detection(model, test_loader, device='cuda'):
    """
    Evaluates the model by calculating reconstruction error for every image.
    High error = Anomaly (Leukemia).
    """
    model.eval()
    criterion = nn.MSELoss(reduction='none') # Loss per pixel
    
    reconstruction_errors = []
    true_labels = []
    
    print("\nEvaluando deteccion de anomalias...")
    
    with torch.no_grad():
        for data, label in test_loader:
            data = data.to(device)
            
            # Reconstruct
            reconstruction = model(data)
            
            # Calculate Error: Mean Squared Error per image
            # loss shape: (Batch, C, H, W) -> mean over dimensions (1,2,3)
            loss = criterion(reconstruction, data)
            loss = loss.mean(dim=(1, 2, 3)) 
            
            reconstruction_errors.extend(loss.cpu().numpy())
            true_labels.extend(label.numpy())
            
    return np.array(reconstruction_errors), np.array(true_labels)

def find_optimal_threshold(errors, labels):
    """
    Finds the threshold that maximizes separation using ROC Curve
    """
    fpr, tpr, thresholds = roc_curve(labels, errors)
    roc_auc = auc(fpr, tpr)
    
    # Optimal threshold: Youden's J statistic (TPR - FPR)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    return optimal_threshold, roc_auc

def plot_results(history, errors, labels, threshold):
    """Plots Training Loss and Anomaly Histograms"""
    plt.figure(figsize=(12, 5))
    
    # 1. Training Loss
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Reconstruction Loss')
    plt.title('Autoencoder Training (Healthy Only)')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    
    # 2. Histogram of Errors
    plt.subplot(1, 2, 2)
    healthy_errors = errors[labels == 0]
    leukemia_errors = errors[labels == 1]
    
    plt.hist(healthy_errors, bins=50, alpha=0.6, label='Healthy (Normal)', color='green')
    plt.hist(leukemia_errors, bins=50, alpha=0.6, label='Leukemia (Anomaly)', color='red')
    plt.axvline(threshold, color='black', linestyle='dashed', linewidth=2, label=f'Threshold: {threshold:.4f}')
    
    plt.title('Reconstruction Error Distribution')
    plt.xlabel('Mean Squared Error (MSE)')
    plt.ylabel('Count')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('autoencoder_results.png')
    plt.show()

# ============ MAIN ============

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Data
    # Assuming cargar_todos_datasets_con_labels works as in CNN.py
    # If not, use the manual loading functions from Carga_imagenes
    print("Cargando datasets...")
    try:
        datasets, _ = cargar_todos_datasets_con_labels()
    except:
        print("Using fallback loading...")
        # Fallback: Load manually if imports fail
        # This part simulates the dict structure expected by prepare_anomaly_data
        datasets = {
            'fold_0_hem': cargar_training_hem_original("data/training_data/fold_0/hem/", max_imagenes=200),
            'fold_0_all': cargar_training_all_original("data/training_data/fold_0/all/", max_imagenes=200)
        }

    # 2. Prepare Data (Train on Healthy, Test on Mixed)
    train_loader, test_loader = prepare_anomaly_data(datasets, batch_size=8)
    
    # 3. Create Model
    model = LeukemiaAutoencoder()
    
    # 4. Train
    model, history = train_autoencoder(model, train_loader, epochs=30, device=device)
    
    # 5. Evaluate
    errors, true_labels = evaluate_anomaly_detection(model, test_loader, device=device)
    
    # 6. Determine Threshold & Metrics
    threshold, roc_auc = find_optimal_threshold(errors, true_labels)
    predicted_labels = (errors > threshold).astype(int)
    
    print("\n" + "="*50)
    print("RESULTADOS FINAL DETECCION DE ANOMALIAS")
    print("="*50)
    print(f"Optimal Threshold (MSE): {threshold:.6f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_labels, target_names=['Healthy', 'Leukemia']))
    
    # 7. Visualize
    plot_results(history, errors, true_labels, threshold)
    
    # Save model
    torch.save(model.state_dict(), 'leukemia_autoencoder.pth')
    print("Modelo guardado como leukemia_autoencoder.pth")

if __name__ == "__main__":
    main()
