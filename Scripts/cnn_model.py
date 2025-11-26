"""
CNN Model for Leukemia Classification
Input: 450x450x3 RGB images (Healthy vs Leukemia cells)
Output: Binary classification
"""

import torch
import torch.nn as nn
import time
import os
import json
from datetime import datetime


class LeukemiaCNN(nn.Module):
    """
    Recommended CNN Architecture for 450x450 images
    
    Architecture reasoning:
    - Input: 450x450x3 → We have high-resolution medical images
    - Strategy: Progressive downsampling with increasing channels
    - Goal: Extract cell morphology features for classification
    """
    
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super(LeukemiaCNN, self).__init__()
        
        # ============ Feature Extraction Blocks ============
        
        # Block 1: 450x450 → 225x225
        # Purpose: Extract low-level features (edges, textures)
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),      # 450x450x32
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
            x: Input tensor of shape (batch_size, 3, 450, 450)
        
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


# ============ MODEL SUMMARY ============
"""
Architecture Summary:
├── Input: (B, 3, 450, 450)
├── Block1: Conv(32) + Conv(32) + MaxPool → (B, 32, 225, 225)
├── Block2: Conv(64) + Conv(64) + MaxPool → (B, 64, 112, 112)
├── Block3: Conv(128) + Conv(128) + MaxPool → (B, 128, 56, 56)
├── Block4: Conv(256) + Conv(256) + MaxPool → (B, 256, 28, 28)
├── AvgPool: (B, 256, 1, 1)
├── Flatten: (B, 256)
└── Classifier: FC(256→512→256→2) → (B, 2)

Total Parameters: ~3.2M
Model Size: ~12-13 MB (float32)

Key Design Decisions:
1. Input size: 450x450 (original image size, no resizing)
2. Progressive downsampling: 4 blocks reduce spatial dims by 2^4 = 16x
3. Channel progression: 32→64→128→256 captures hierarchical features
4. Batch Norm + ReLU after each conv: Stabilizes training
5. MaxPool: Reduces computation, adds translation invariance
6. Dropout in classifier: Prevents overfitting
7. AdaptiveAvgPool: Works with any spatial dimensions
"""


# ============ COMPUTATIONAL ANALYSIS FUNCTIONS ============

def contar_parametros_modelo(model):
    """
    Cuenta el número total de parámetros del modelo
    
    Args:
        model: Modelo PyTorch
    
    Returns:
        dict: Información de parámetros
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total_params - trainable_params
    
    # Estimar tamaño en MB (float32 = 4 bytes por parámetro)
    model_size_mb = (total_params * 4) / (1024 ** 2)
    
    return {
        'total_parametros': total_params,
        'parametros_entrenables': trainable_params,
        'parametros_no_entrenables': non_trainable,
        'tamaño_modelo_mb': model_size_mb
    }


def estimar_flops_inferencia(model, input_shape=(1, 3, 450, 450)):
    """
    Estima FLOPs (Floating Point Operations) para una pasada forward
    
    Args:
        model: Modelo PyTorch
        input_shape: Forma del input (batch, canales, alto, ancho)
    
    Returns:
        dict: Estimación de FLOPs
    """
    # Estimación simplificada: convoluciones son la mayoría de operaciones
    flops = 0
    
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            # FLOPs = 2 * kernel_size * kernel_size * in_channels * out_height * out_width * batch_size
            kernel_ops = module.kernel_size[0] * module.kernel_size[1]
            output_size = (input_shape[0], module.out_channels, 
                          input_shape[2] // 2, input_shape[3] // 2)  # Aproximado
            flops += 2 * kernel_ops * module.in_channels * output_size[2] * output_size[3] * output_size[0]
    
    return {
        'flops_estimados': flops,
        'flops_giga': flops / 1e9,
        'nota': 'Estimación basada en operaciones de convolución'
    }


def medir_tiempo_inferencia(model, batch_size=32, num_iterations=100, prefijo="cnn"):
    """
    Mide el tiempo de inferencia del modelo
    
    Args:
        model: Modelo PyTorch (debe estar en GPU o CPU)
        batch_size: Tamaño del batch
        num_iterations: Número de iteraciones para medir
        prefijo: Prefijo para los datos (ej: "cnn")
    
    Returns:
        dict: Métricas de tiempo
    """
    device = next(model.parameters()).device
    model.eval()
    
    # Input dummy
    dummy_input = torch.randn(batch_size, 3, 450, 450).to(device)
    
    # Warmup (calentamiento)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Sincronizar si está en GPU
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Medición
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    # Calcular métricas
    total_time = end_time - start_time
    avg_time_ms = (total_time / num_iterations) * 1000
    throughput = (num_iterations * batch_size) / total_time
    
    metricas_tiempo = {
        'prefijo': prefijo,
        'batch_size': batch_size,
        'num_iteraciones': num_iterations,
        'tiempo_total_segundos': total_time,
        'tiempo_promedio_ms': avg_time_ms,
        'throughput_imagenes_por_segundo': throughput,
        'dispositivo': str(device)
    }
    
    return metricas_tiempo


def guardar_metricas_computacionales(model, output_path="logs", prefijo="cnn", batch_size=32):
    """
    Calcula y guarda todas las métricas computacionales en un archivo de texto
    Similar a guardar_imagenes_reconstruidas en pca_flops.py
    
    Args:
        model: Modelo PyTorch
        output_path: Ruta donde guardar las métricas
        prefijo: Prefijo para el nombre del archivo (ej: "cnn")
        batch_size: Tamaño del batch para mediciones
    
    Returns:
        dict: Todas las métricas compiladas
    """
    # Crear directorio si no existe
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    print(f"\nCalculando métricas computacionales del modelo CNN...")
    
    # Recopilar todas las métricas
    all_metrics = {}
    
    # 1. Parámetros
    print("  - Contando parámetros...")
    all_metrics['parametros'] = contar_parametros_modelo(model)
    
    # 2. FLOPs estimados
    print("  - Estimando FLOPs...")
    all_metrics['flops'] = estimar_flops_inferencia(model, input_shape=(batch_size, 3, 450, 450))
    
    # 3. Tiempo de inferencia
    print("  - Midiendo tiempo de inferencia...")
    all_metrics['tiempo_inferencia'] = medir_tiempo_inferencia(model, batch_size=batch_size, prefijo=prefijo)
    
    # 4. Información general
    all_metrics['informacion_general'] = {
        'timestamp': datetime.now().isoformat(),
        'modelo': 'LeukemiaCNN',
        'tamaño_entrada': [batch_size, 3, 450, 450],
        'clases': 2
    }
    
    # Generar reporte en texto
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefijo}_metricas_computacionales_{timestamp}.txt"
    filepath = os.path.join(output_path, filename)
    
    with open(filepath, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("ANÁLISIS COMPUTACIONAL - CNN LEUKEMIA CLASSIFICATION\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Timestamp: {all_metrics['informacion_general']['timestamp']}\n")
        f.write(f"Modelo: {all_metrics['informacion_general']['modelo']}\n")
        f.write(f"Dispositivo: {all_metrics['tiempo_inferencia']['dispositivo']}\n\n")
        
        # Parámetros
        f.write("-" * 70 + "\n")
        f.write("PARÁMETROS DEL MODELO\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total de parámetros: {all_metrics['parametros']['total_parametros']:,}\n")
        f.write(f"Parámetros entrenables: {all_metrics['parametros']['parametros_entrenables']:,}\n")
        f.write(f"Parámetros no entrenables: {all_metrics['parametros']['parametros_no_entrenables']:,}\n")
        f.write(f"Tamaño del modelo: {all_metrics['parametros']['tamaño_modelo_mb']:.2f} MB\n\n")
        
        # FLOPs
        f.write("-" * 70 + "\n")
        f.write("ANÁLISIS DE FLOPS (Operaciones de punto flotante)\n")
        f.write("-" * 70 + "\n")
        f.write(f"FLOPs estimados por batch: {all_metrics['flops']['flops_estimados']:,.0f}\n")
        f.write(f"FLOPs en GigaFLOPs: {all_metrics['flops']['flops_giga']:.4f}\n")
        f.write(f"Nota: {all_metrics['flops']['nota']}\n\n")
        
        # Tiempo de inferencia
        f.write("-" * 70 + "\n")
        f.write("TIEMPO DE INFERENCIA\n")
        f.write("-" * 70 + "\n")
        f.write(f"Batch size: {all_metrics['tiempo_inferencia']['batch_size']}\n")
        f.write(f"Número de iteraciones: {all_metrics['tiempo_inferencia']['num_iteraciones']}\n")
        f.write(f"Tiempo total: {all_metrics['tiempo_inferencia']['tiempo_total_segundos']:.4f} segundos\n")
        f.write(f"Tiempo promedio por batch: {all_metrics['tiempo_inferencia']['tiempo_promedio_ms']:.2f} ms\n")
        f.write(f"Throughput: {all_metrics['tiempo_inferencia']['throughput_imagenes_por_segundo']:.2f} imágenes/segundo\n\n")
        
        # Resumen
        f.write("-" * 70 + "\n")
        f.write("RESUMEN DE RENDIMIENTO\n")
        f.write("-" * 70 + "\n")
        f.write(f"Tamaño del modelo: {all_metrics['parametros']['tamaño_modelo_mb']:.2f} MB\n")
        f.write(f"Latencia promedio: {all_metrics['tiempo_inferencia']['tiempo_promedio_ms']:.2f} ms\n")
        f.write(f"Throughput: {all_metrics['tiempo_inferencia']['throughput_imagenes_por_segundo']:.2f} img/s\n")
        f.write("=" * 70 + "\n")
    
    # También guardar como JSON para referencia
    json_filename = f"{prefijo}_metricas_computacionales_{timestamp}.json"
    json_filepath = os.path.join(output_path, json_filename)
    
    with open(json_filepath, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    
    print(f"\n✓ Métricas guardadas en:")
    print(f"  - Texto: {filepath}")
    print(f"  - JSON: {json_filepath}")
    print(f"  - Ruta: {os.path.abspath(output_path)}\n")
    
    return all_metrics


if __name__ == "__main__":
    # Quick test
    model = LeukemiaCNN(num_classes=2)
    print(model)
    
    # Test forward pass
    x = torch.randn(4, 3, 450, 450)  # Batch of 4 images
    output = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Ejemplo: Guardar métricas computacionales
    print("\n" + "=" * 70)
    print("ANÁLISIS COMPUTACIONAL")
    print("=" * 70)
    metricas = guardar_metricas_computacionales(model, output_path="logs", prefijo="cnn", batch_size=32)
