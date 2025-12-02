"""
Autoencoder Variacional (VAE) Mejorado para Detección de Leucemia
Incluye: Diagnóstico por época, visualizaciones y análisis avanzado
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
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split
import cv2
import seaborn as sns
from scipy import stats

# ============ VAE AUTOENCODER MEJORADO ============

class LeukemiaVAE(nn.Module):
    """
    Variational Autoencoder con bottleneck controlado para 450x450 imágenes
    """
    def __init__(self, latent_dim=64):
        super(LeukemiaVAE, self).__init__()
        self.latent_dim = latent_dim
        
        # ============ ENCODER ============
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # REDUCIDO de 32
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # REDUCIDO de 64
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # REDUCIDO de 128
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # REDUCIDO de 256
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2)  # 450→28×28
        )
        
        # Flatten y bottleneck VARIACIONAL
        self.flatten_size = 128 * 28 * 28  # 128×28×28 = 100,352
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_var = nn.Linear(self.flatten_size, latent_dim)
        
        # Decoder input
        self.fc_decode = nn.Linear(latent_dim, self.flatten_size)
        
        # ============ DECODER ============
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.Upsample(size=(112, 112), mode='bilinear', align_corners=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            
            nn.Upsample(size=(450, 450), mode='bilinear', align_corners=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        z = self.fc_decode(z)
        z = z.view(-1, 128, 28, 28)
        return self.decoder(z)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z)
        return reconstruction, mu, log_var, z
    
    def reconstruct(self, x):
        """Solo reconstrucción (sin sampling)"""
        mu, _ = self.encode(x)
        return self.decode(mu)

# ============ FUNCIONES DIAGNÓSTICO AVANZADO ============

class VAE_Diagnostics:
    """Herramientas para diagnóstico durante entrenamiento"""
    
    @staticmethod
    def compute_vae_loss(recon_x, x, mu, log_var, beta=0.1):
        """Loss VAE: Reconstruction + KL Divergence"""
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + beta * kl_loss, recon_loss, kl_loss
    
    @staticmethod
    def analyze_latent_space(z_batch, labels=None, epoch=None):
        """Análisis del espacio latente por época"""
        z_np = z_batch.cpu().numpy()
        
        stats_dict = {
            'mean_norm': np.mean(np.linalg.norm(z_np, axis=1)),
            'std_norm': np.std(np.linalg.norm(z_np, axis=1)),
            'latent_mean': np.mean(z_np, axis=0).tolist(),
            'latent_std': np.std(z_np, axis=0).tolist()
        }
        
        if labels is not None:
            # Separabilidad entre clases
            healthy_z = z_np[labels == 0]
            leukemia_z = z_np[labels == 1]
            if len(healthy_z) > 0 and len(leukemia_z) > 0:
                stats_dict['class_separation'] = float(
                    np.linalg.norm(np.mean(healthy_z, axis=0) - np.mean(leukemia_z, axis=0))
                )
        
        return stats_dict
    
    @staticmethod
    def compute_reconstruction_metrics(original, reconstructed):
        """Métricas detalladas de reconstrucción"""
        original_np = original.cpu().numpy().flatten()
        recon_np = reconstructed.cpu().numpy().flatten()
        
        mse = np.mean((original_np - recon_np) ** 2)
        mae = np.mean(np.abs(original_np - recon_np))
        ssim_value = VAE_Diagnostics._compute_ssim(original_np, recon_np)
        
        return {
            'mse': float(mse),
            'mae': float(mae),
            'ssim': float(ssim_value),
            'psnr': float(10 * np.log10(1.0 / mse)) if mse > 0 else 100.0
        }
    
    @staticmethod
    def _compute_ssim(x, y):
        """Simplified SSIM (para diagnóstico)"""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        mu_x = np.mean(x)
        mu_y = np.mean(y)
        sigma_x = np.var(x)
        sigma_y = np.var(y)
        sigma_xy = np.cov(x, y)[0, 1]
        
        numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
        
        return numerator / denominator if denominator != 0 else 0

# ============ TRAINING CON DIAGNÓSTICO POR ÉPOCA ============

def train_vae_with_diagnostics(model, train_loader, test_data, epochs=25, lr=0.001, 
                              beta=0.1, device='cuda', save_dir="vae_diagnostics"):
    """Entrenamiento VAE con diagnóstico completo por época"""
    
    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Métricas por época
    epoch_metrics = {
        'train_loss': [], 'recon_loss': [], 'kl_loss': [],
        'val_recon_metrics': [], 'latent_stats': [],
        'epoch_times': []
    }
    
    # Datos de test para diagnóstico
    X_test_tensor, y_test = test_data
    test_subset = X_test_tensor[:32].to(device)  # Solo 32 para diagnóstico rápido
    test_labels = y_test[:32]
    
    print(f"\n{'='*60}")
    print("ENTRENAMIENTO VAE CON DIAGNÓSTICO POR ÉPOCA")
    print(f"{'='*60}\n")
    
    header = f"{'Epoch':^6} | {'Total Loss':^12} | {'Recon Loss':^12} | {'KL Loss':^10} | {'Val MSE':^10} | {'Latent Sep':^10} | {'Time':^8}"
    print(header)
    print("-" * 90)
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # ===== TRAINING =====
        model.train()
        epoch_train_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            
            optimizer.zero_grad()
            reconstruction, mu, log_var, z = model(data)
            
            total_loss, recon_loss, kl_loss = VAE_Diagnostics.compute_vae_loss(
                reconstruction, data, mu, log_var, beta=beta
            )
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_train_loss += total_loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()
        
        # ===== DIAGNÓSTICO POR ÉPOCA =====
        model.eval()
        with torch.no_grad():
            # 1. Reconstrucción en validation
            val_reconstruction = model.reconstruct(test_subset)
            recon_metrics = VAE_Diagnostics.compute_reconstruction_metrics(
                test_subset, val_reconstruction
            )
            
            # 2. Análisis del espacio latente
            _, _, _, z_latent = model(test_subset)
            latent_stats = VAE_Diagnostics.analyze_latent_space(
                z_latent, test_labels, epoch=epoch
            )
            
            # 3. Guardar visualizaciones cada 5 épocas
            if (epoch + 1) % 5 == 0:
                save_epoch_visualizations(
                    model, test_subset, test_labels, epoch, save_dir
                )
        
        # ===== LOGGING =====
        epoch_time = time.time() - epoch_start
        
        epoch_metrics['train_loss'].append(epoch_train_loss / len(train_loader))
        epoch_metrics['recon_loss'].append(epoch_recon_loss / len(train_loader))
        epoch_metrics['kl_loss'].append(epoch_kl_loss / len(train_loader))
        epoch_metrics['val_recon_metrics'].append(recon_metrics)
        epoch_metrics['latent_stats'].append(latent_stats)
        epoch_metrics['epoch_times'].append(epoch_time)
        
        # Print formato bonito
        latent_sep = latent_stats.get('class_separation', 0.0)
        print(f"{epoch+1:^6} | "
              f"{epoch_metrics['train_loss'][-1]:^12.4f} | "
              f"{epoch_metrics['recon_loss'][-1]:^12.4f} | "
              f"{epoch_metrics['kl_loss'][-1]:^10.4f} | "
              f"{recon_metrics['mse']:^10.6f} | "
              f"{latent_sep:^10.4f} | "
              f"{epoch_time:^8.2f}")
    
    print(f"\n{'='*90}")
    print("ENTRENAMIENTO COMPLETADO - GENERANDO REPORTES FINALES")
    print(f"{'='*90}")
    
    # Generar reporte final
    generate_final_diagnostics(model, epoch_metrics, test_data, save_dir, device)
    
    return model, epoch_metrics

# ============ VISUALIZACIONES POR ÉPOCA ============

def save_epoch_visualizations(model, test_data, test_labels, epoch, save_dir):
    """Guarda visualizaciones cada 5 épocas"""
    model.eval()
    with torch.no_grad():
        reconstructions, mu, log_var, z = model(test_data)
        
        fig, axes = plt.subplots(4, 8, figsize=(20, 10))
        
        # Mostrar originales y reconstrucciones
        for i in range(8):
            # Original
            axes[0, i].imshow(test_data[i, 0].cpu().numpy(), cmap='gray')
            axes[0, i].set_title(f"Original\nLabel: {test_labels[i]}")
            axes[0, i].axis('off')
            
            # Reconstrucción
            axes[1, i].imshow(reconstructions[i, 0].cpu().numpy(), cmap='gray')
            axes[1, i].set_title(f"Reconstructed\nEpoch {epoch+1}")
            axes[1, i].axis('off')
            
            # Diferencia
            diff = np.abs(test_data[i, 0].cpu().numpy() - 
                         reconstructions[i, 0].cpu().numpy())
            axes[2, i].imshow(diff, cmap='hot')
            axes[2, i].set_title(f"Diff (×10)\nMSE: {diff.mean():.6f}")
            axes[2, i].axis('off')
            
            # Histograma de diferencia
            axes[3, i].hist(diff.flatten(), bins=50, alpha=0.7)
            axes[3, i].set_title(f"Diff Histogram")
            axes[3, i].set_xlim(0, 0.1)
        
        plt.suptitle(f'VAE Reconstruction - Epoch {epoch+1}', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'epoch_{epoch+1:03d}_reconstruction.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # Visualización espacio latente (si 2D o 3D)
        if model.latent_dim in [2, 3]:
            save_latent_visualization(z, test_labels, epoch, save_dir)

def save_latent_visualization(z, labels, epoch, save_dir):
    """Visualización del espacio latente"""
    z_np = z.cpu().numpy()
    
    if z_np.shape[1] == 2:
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(z_np[:, 0], z_np[:, 1], c=labels, 
                             cmap='coolwarm', alpha=0.6, edgecolors='k')
        plt.colorbar(scatter, label='Label (0=Healthy, 1=Leukemia)')
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('Latent Dimension 2')
        plt.title(f'Latent Space Visualization - Epoch {epoch+1}')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, f'epoch_{epoch+1:03d}_latent_2d.png'), 
                   dpi=150)
        plt.close()

# ============ REPORTE FINAL MEJORADO ============

def generate_final_diagnostics(model, epoch_metrics, test_data, save_dir, device):
    """Genera reporte diagnóstico completo"""
    
    X_test_tensor, y_test = test_data
    X_test_tensor = X_test_tensor.to(device)
    
    # 1. Evaluación final
    model.eval()
    with torch.no_grad():
        errors = []
        latent_codes = []
        
        for i in range(0, len(X_test_tensor), 32):
            batch = X_test_tensor[i:i+32]
            reconstruction = model.reconstruct(batch)
            batch_errors = nn.functional.mse_loss(
                reconstruction, batch, reduction='none'
            ).mean(dim=(1,2,3)).cpu().numpy()
            errors.extend(batch_errors)
            
            # Obtener códigos latentes
            mu, _ = model.encode(batch)
            latent_codes.extend(mu.cpu().numpy())
    
    errors = np.array(errors)
    latent_codes = np.array(latent_codes)
    
    # 2. Análisis de errores por clase
    healthy_errors = errors[y_test == 0]
    leukemia_errors = errors[y_test == 1]
    
    error_stats = {
        'healthy_mean': float(np.mean(healthy_errors)),
        'healthy_std': float(np.std(healthy_errors)),
        'leukemia_mean': float(np.mean(leukemia_errors)),
        'leukemia_std': float(np.std(leukemia_errors)),
        'effect_size': float((np.mean(leukemia_errors) - np.mean(healthy_errors)) / 
                           np.sqrt((np.std(healthy_errors)**2 + np.std(leukemia_errors)**2)/2)),
        'overlap_percentage': float(np.sum(healthy_errors > np.percentile(leukemia_errors, 25)) / 
                                  len(healthy_errors) * 100)
    }
    
    # 3. Encontrar threshold óptimo
    fpr, tpr, thresholds = roc_curve(y_test, errors)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    roc_auc = auc(fpr, tpr)
    
    # Threshold por percentil
    percentile_95_threshold = np.percentile(healthy_errors, 95)
    percentile_99_threshold = np.percentile(healthy_errors, 99)
    
    # 4. Métricas con diferentes thresholds
    metrics_per_threshold = {}
    for thresh_name, threshold in [
        ('optimal', optimal_threshold),
        ('p95', percentile_95_threshold),
        ('p99', percentile_99_threshold)
    ]:
        preds = (errors > threshold).astype(int)
        report = classification_report(y_test, preds, 
                                      target_names=['Healthy', 'Leukemia'],
                                      output_dict=True)
        metrics_per_threshold[thresh_name] = {
            'threshold': float(threshold),
            'recall_leukemia': report['Leukemia']['recall'],
            'precision_leukemia': report['Leukemia']['precision'],
            'recall_healthy': report['Healthy']['recall'],
            'f1_leukemia': report['Leukemia']['f1-score']
        }
    
    # 5. Generar reporte completo
    report = {
        'training_summary': {
            'final_train_loss': epoch_metrics['train_loss'][-1],
            'final_recon_loss': epoch_metrics['recon_loss'][-1],
            'final_kl_loss': epoch_metrics['kl_loss'][-1],
            'total_epochs': len(epoch_metrics['train_loss']),
            'total_time': sum(epoch_metrics['epoch_times'])
        },
        'error_analysis': error_stats,
        'threshold_analysis': {
            'roc_auc': float(roc_auc),
            'optimal_threshold': float(optimal_threshold),
            'threshold_metrics': metrics_per_threshold
        },
        'latent_space_analysis': {
            'dimension': model.latent_dim,
            'latent_norm_mean': float(np.mean(np.linalg.norm(latent_codes, axis=1))),
            'latent_norm_std': float(np.std(np.linalg.norm(latent_codes, axis=1)))
        }
    }
    
    # 6. Guardar reporte
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(save_dir, f'vae_diagnostics_report_{timestamp}.json')
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    # 7. Plot final
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Error distributions
    plt.subplot(2, 3, 1)
    plt.hist(healthy_errors, bins=50, alpha=0.7, label='Healthy', density=True)
    plt.hist(leukemia_errors, bins=50, alpha=0.7, label='Leukemia', density=True)
    plt.axvline(optimal_threshold, color='r', linestyle='--', label=f'Threshold: {optimal_threshold:.6f}')
    plt.xlabel('Reconstruction Error (MSE)')
    plt.ylabel('Density')
    plt.title('Error Distributions by Class')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: ROC Curve
    plt.subplot(2, 3, 2)
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Training Loss
    plt.subplot(2, 3, 3)
    plt.plot(epoch_metrics['train_loss'], label='Total Loss')
    plt.plot(epoch_metrics['recon_loss'], label='Recon Loss')
    plt.plot(epoch_metrics['kl_loss'], label='KL Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Latent separation over time
    plt.subplot(2, 3, 4)
    latent_seps = [s.get('class_separation', 0) for s in epoch_metrics['latent_stats']]
    plt.plot(range(1, len(latent_seps)+1), latent_seps)
    plt.xlabel('Epoch')
    plt.ylabel('Latent Class Separation')
    plt.title('Latent Space Evolution')
    plt.grid(True, alpha=0.3)
    
    # Subplot 5: Validation MSE over time
    plt.subplot(2, 3, 5)
    val_mses = [m['mse'] for m in epoch_metrics['val_recon_metrics']]
    plt.plot(range(1, len(val_mses)+1), val_mses)
    plt.xlabel('Epoch')
    plt.ylabel('Validation MSE')
    plt.title('Validation Reconstruction Error')
    plt.grid(True, alpha=0.3)
    
    # Subplot 6: Metrics comparison
    plt.subplot(2, 3, 6)
    thresholds_names = list(metrics_per_threshold.keys())
    recalls = [metrics_per_threshold[t]['recall_leukemia'] for t in thresholds_names]
    precisions = [metrics_per_threshold[t]['precision_leukemia'] for t in thresholds_names]
    
    x = np.arange(len(thresholds_names))
    width = 0.35
    
    plt.bar(x - width/2, recalls, width, label='Recall Leukemia', alpha=0.8)
    plt.bar(x + width/2, precisions, width, label='Precision Leukemia', alpha=0.8)
    
    plt.xlabel('Threshold Method')
    plt.ylabel('Score')
    plt.title('Performance by Threshold')
    plt.xticks(x, thresholds_names)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'VAE Diagnostics Report - {timestamp}', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'final_diagnostics_{timestamp}.png'), 
               dpi=150, bbox_inches='tight')
    plt.close()
    
    # 8. Print summary
    print(f"\n{'='*60}")
    print("RESUMEN FINAL - VAE DIAGNÓSTICO")
    print(f"{'='*60}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Error Healthy: {error_stats['healthy_mean']:.6f} ± {error_stats['healthy_std']:.6f}")
    print(f"Error Leukemia: {error_stats['leukemia_mean']:.6f} ± {error_stats['leukemia_std']:.6f}")
    print(f"Effect Size: {error_stats['effect_size']:.2f}")
    print(f"Overlap (FP risk): {error_stats['overlap_percentage']:.1f}%")
    print(f"\nThreshold Recommendations:")
    for thresh_name, metrics in metrics_per_threshold.items():
        print(f"  {thresh_name}: Recall={metrics['recall_leukemia']:.3f}, "
              f"Precision={metrics['precision_leukemia']:.3f}, "
              f"Threshold={metrics['threshold']:.6f}")
    print(f"\nReport saved to: {report_path}")

# ============ MAIN UPDATED ============

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Data (usando tus funciones existentes)
    print("Loading datasets...")
    try:
        from Creador_labels import cargar_todos_datasets_con_labels
        datasets, _ = cargar_todos_datasets_con_labels()
    except:
        print("Using fallback loading...")
        # Aquí tu código de carga existente
    
    # 2. Prepare Data (igual que antes)
    def prepare_anomaly_data(datasets, batch_size=16):
        # Tu código existente aquí
        pass
    
    train_loader, test_loader, (X_train_tensor, X_test_tensor, y_test) = prepare_anomaly_data(
        datasets, batch_size=16
    )
    
    # 3. Create VAE Model
    print(f"\nCreating VAE model with latent_dim=64...")
    model = LeukemiaVAE(latent_dim=64)
    
    # 4. Train with diagnostics
    model, metrics = train_vae_with_diagnostics(
        model=model,
        train_loader=train_loader,
        test_data=(X_test_tensor, y_test),
        epochs=25,
        lr=0.001,
        beta=0.1,  # Peso de KL divergence
        device=device,
        save_dir="vae_detailed_logs"
    )
    
    # 5. Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'metrics': metrics,
        'latent_dim': model.latent_dim
    }, 'leukemia_vae_improved.pth')
    
    print(f"\nModel saved as 'leukemia_vae_improved.pth'")
    print(f"Diagnostic logs saved in 'vae_detailed_logs/' directory")

if __name__ == "__main__":
    main()