"""

Importe de librerias

"""
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

"""
Definicion de funciones para aplicacion de pca y guardado de imagenes

"""

def aplicar_pca(imagenes, sample_size=500, componentes=50):
    """
    Aplica PCA para reducción de dimensionalidad y reconstrucción de imágenes
    
    Args:
        imagenes: Lista de imágenes originales
        sample_size: Número de imágenes a procesar
        componentes: Número de componentes principales
    
    Returns:
        tuple: (imagenes_reconstruidas, imagenes_gray, shape_original)
    """
    # Seleccionar muestra
    muestra = imagenes[:sample_size]
    
    # 1. Convertir a numpy array
    imagenes_np = np.array(muestra)
    n_imgs, h, w, c = imagenes_np.shape
    
    # 2. Convertir a escala de grises
    if c == 3:
        imagenes_gray = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imagenes_np])
    else:
        imagenes_gray = imagenes_np.squeeze()
    
    # 3. Aplanar cada imagen
    X = imagenes_gray.reshape(n_imgs, -1)
    
    # 4. Centrar los datos
    X_bar = X.mean(axis=0)
    X_centered = X - X_bar
    
    # 5. PCA usando SVD
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    
    # 6. Seleccionar componentes principales
    E = Vt[:componentes].T
    
    # 7. Proyectar y reconstruir
    Y = X_centered @ E
    X_hat = (Y @ E.T) + X_bar
    
    # 8. Volver a la forma original
    imagenes_reconstruidas = X_hat.reshape(n_imgs, h, w)
    
    return imagenes_reconstruidas, imagenes_gray, (h, w)

def visualizar_reconstruccion(original, reconstruida, titulo="PCA"):
    """
    Visualiza comparación entre imágenes originales y reconstruidas
    
    Args:
        original: Imágenes originales en escala de grises
        reconstruida: Imágenes reconstruidas
        titulo: Título para la visualización
    """
    n_imgs = len(original)
    
    plt.figure(figsize=(12, 6))
    for i in range(min(6, n_imgs)):
        # Imagen original
        plt.subplot(2, 6, i+1)
        plt.imshow(original[i], cmap='gray')
        plt.title("Original")
        plt.axis('off')
        
        # Imagen reconstruida
        plt.subplot(2, 6, i+7)
        plt.imshow(reconstruida[i], cmap='gray')
        plt.title("Reconstruida")
        plt.axis('off')
    
    plt.suptitle(titulo)
    plt.tight_layout()
    plt.show()

def guardar_imagenes_reconstruidas(imagenes_reconstruidas, output_path, prefijo="imagen"):
    """
    Guarda las imágenes reconstruidas en disco
    
    Args:
        imagenes_reconstruidas: Imágenes reconstruidas
        output_path: Ruta donde guardar las imágenes
        prefijo: Prefijo para los nombres de archivo
    """
    # Crear directorio si no existe
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Convertir a uint8
    imagenes_uint8 = np.clip(imagenes_reconstruidas, 0, 255).astype(np.uint8)
    
    # Guardar cada imagen
    saved_count = 0
    for i in range(len(imagenes_uint8)):
        filename = f"{prefijo}_{i+1}.bmp"
        filepath = os.path.join(output_path, filename)
        success = cv2.imwrite(filepath, imagenes_uint8[i])
        
        if success:
            saved_count += 1
            if i % 50 == 0:
                print(f"Guardada imagen {i+1}/{len(imagenes_uint8)}")
        else:
            print(f"Error al guardar imagen {i+1}")
    
    print(f"\nProceso completado!")
    print(f"Total de imágenes guardadas: {saved_count}")
    print(f"Ruta: {os.path.abspath(output_path)}")
    
    return saved_count

"""

Version mas con todo incluido

"""

def procesar_pca_completo(imagenes, output_path, prefijo_archivo, 
                         sample_size=500, componentes=50, nombre_dataset=""):
    """
    Función completa que aplica PCA, visualiza y guarda resultados
    
    Args:
        imagenes: Lista de imágenes originales
        output_path: Ruta para guardar imágenes reconstruidas
        prefijo_archivo: Prefijo para nombres de archivo
        sample_size: Tamaño de la muestra
        componentes: Número de componentes PCA
        nombre_dataset: Nombre del dataset para visualización
    """
    print(f"Procesando {nombre_dataset}...")
    
    # Aplicar PCA
    imagenes_reconstruidas, imagenes_original_gray, shape = aplicar_pca(
        imagenes, sample_size, componentes
    )
    
    # Visualizar resultados
    visualizar_reconstruccion(
        imagenes_original_gray, 
        imagenes_reconstruidas, 
        f"PCA - {nombre_dataset} ({componentes} componentes)"
    )
    
    # Guardar imágenes
    guardar_imagenes_reconstruidas(
        imagenes_reconstruidas, 
        output_path, 
        prefijo_archivo
    )
    
    return imagenes_reconstruidas