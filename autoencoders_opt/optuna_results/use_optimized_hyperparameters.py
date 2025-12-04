"""
HOW TO USE THE OPTUNA-OPTIMIZED HYPERPARAMETERS
Generated: 2025-12-03 22:36:16
Best ROC AUC: 0.7472
"""

import torch
import torch.nn as nn

class OriginalAutoencoder(nn.Module):
    def __init__(self, 
                 channels_multiplier=0.5,
                 use_batchnorm=True,
                 use_dropout=True,
                 dropout_rate=0.25):
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
    hyperparams = {
        # Architecture hyperparameters
        'channels_multiplier': 0.5,
        'use_batchnorm': True,
        'use_dropout': True,
        'dropout_rate': 0.25,
        
        # Training hyperparameters
        'learning_rate': 0.0031404382679661026,
        'weight_decay': 0.00011554231483529049,
        'optimizer': 'adamw',
        'epochs': 14,
        'noise_level': 0.125,
    }
    
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
        print(f"  {key}: {value}")
    
    # Create model with optimized architecture
    model = create_optimized_model()
    print(f"\nModel created with {sum(p.numel() for p in model.parameters()):,} parameters")
