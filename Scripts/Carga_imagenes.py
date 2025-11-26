
""" Importe de librerias"""


import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

""" Definicion de funciones para cargar y preprocesar imágenes. """

def cargar_imagenes_secuenciales(ruta, prefijo="", max_imagenes=500, inicio=1, fin=3000):
    """
    Carga imágenes con nombres secuenciales como:
    - Original: "1.bmp", "2.bmp", ...
    - Reconstruida: "testing_1.bmp", "validation_1.bmp", etc.
    
    Args:
        ruta: Directorio donde están las imágenes
        prefijo: Prefijo del nombre (ej: "testing_", "validation_")
        max_imagenes: Máximo número de imágenes a cargar
        inicio: Número inicial para buscar
        fin: Número final para buscar
    
    Returns:
        Lista de imágenes cargadas
    """
    imagenes = []
    for i in range(inicio, fin + 1):
        if len(imagenes) >= max_imagenes:
            break
        nombre_archivo = os.path.join(ruta, f"{prefijo}{i}.bmp")
        if not os.path.exists(nombre_archivo):
            continue
        img = cv2.imread(nombre_archivo)
        if img is not None:
            imagenes.append(img)
            if len(imagenes) % 50 == 0:
                print(f"Cargadas {len(imagenes)}/{max_imagenes} imágenes")
        else:
            print(f"Advertencia: no se pudo cargar {nombre_archivo}")
    
    print(f"Total imágenes cargadas: {len(imagenes)}")
    return imagenes


def cargar_training_all_original(ruta, max_imagenes=500, limites_busqueda=None):
    """
    Carga training data original con patrón UID_x_y_z_all.bmp
    Busca automáticamente en rangos amplios hasta encontrar las imágenes.
    
    Args:
        ruta: Directorio del fold (ej: "data/training_data/fold_0/all/")
        max_imagenes: Máximo de imágenes a cargar
        limites_busqueda: Tupla con (max_x, max_y, max_z). Si es None, usa valores por defecto grandes.
    
    Returns:
        Lista de imágenes cargadas
    """
    if limites_busqueda is None:
        # Rangos amplios por defecto para buscar automáticamente
        max_x, max_y, max_z = 250, 250, 50
    else:
        max_x, max_y, max_z = limites_busqueda
    
    imagenes = []
    total_loaded = 0
    
    print(f"Buscando imágenes en {ruta}...")
    
    for x in range(1, max_x + 1):
        if total_loaded >= max_imagenes:
            break
        found_any_for_x = False
        for y in range(1, max_y + 1):
            if total_loaded >= max_imagenes:
                break
            found_any_for_y = False
            for z in range(1, max_z + 1):
                if total_loaded >= max_imagenes:
                    break
                filename = f"UID_{x}_{y}_{z}_all.bmp"
                filepath = os.path.join(ruta, filename)
                if os.path.exists(filepath):
                    img = cv2.imread(filepath)
                    if img is not None:
                        imagenes.append(img)
                        total_loaded += 1
                        found_any_for_y = True
                        found_any_for_x = True
                        if total_loaded % 50 == 0:
                            print(f"Cargado: {filename} (Total: {total_loaded}/{max_imagenes})")
                # Si no existe el archivo, continuar con el siguiente z
            # Si no se encontró ninguna imagen para este y, continuar al siguiente y
            if not found_any_for_y:
                continue
        # Si no se encontró ninguna imagen para este x, pasar al siguiente x
        if not found_any_for_x:
            continue
    
    print(f"Total imágenes cargadas: {total_loaded}")
    return imagenes

def cargar_training_hem_original(ruta, max_imagenes=500, start_x=1, end_x=49, max_y=30, max_z=20):
    """
    Carga training data con patrón UID_HX_Y_Z_hem.bmp
    Basado en el código que funciona con la estructura real de archivos.
    
    Args:
        ruta: Directorio del fold (ej: "data/training_data/fold_2/hem/")
        max_imagenes: Máximo de imágenes a cargar
        start_x: Valor inicial de x (H1, H2, etc.)
        end_x: Valor final de x
        max_y: Máximo valor de y
        max_z: Máximo valor de z
    
    Returns:
        Lista de imágenes cargadas
    """
    imagenes = []
    total_loaded = 0
    
    print(f"Buscando imágenes en {ruta}...")
    
    for x in range(start_x, end_x + 1):
        if total_loaded >= max_imagenes:
            break
        found_any_for_x = False
        for y in range(1, max_y + 1):
            if total_loaded >= max_imagenes:
                break
            found_any_for_y = False
            for z in range(1, max_z + 1):
                if total_loaded >= max_imagenes:
                    break
                filename = f"UID_H{x}_{y}_{z}_hem.bmp"
                filepath = os.path.join(ruta, filename)
                if os.path.exists(filepath):
                    img = cv2.imread(filepath)
                    if img is not None:
                        imagenes.append(img)
                        total_loaded += 1
                        found_any_for_y = True
                        found_any_for_x = True
                        print(f"Cargado: {filename} (Total cargados: {total_loaded})")
                    else:
                        print(f"Advertencia: no se pudo cargar {filename}")
                # Si no existe el archivo, continúa para probar las demás combinaciones
            # Si no se encontró ninguna imagen para este y, saltamos al siguiente y
            if not found_any_for_y:
                continue
        # Si no se encontró ninguna imagen en todo y,z para este x, saltar a siguiente x
        if not found_any_for_x:
            print(f"No hay imágenes para x=H{x}, se pasa al siguiente.")
            continue

    print(f"Carga finalizada. Total imágenes cargadas: {total_loaded}")
    return imagenes


def visualizar_imagenes(imagenes, titulo="Imágenes", filas=2, columnas=3):
    """
    Visualiza imágenes en una grid
    
    Args:
        imagenes: Lista de imágenes
        titulo: Título para la visualización
        filas: Número de filas
        columnas: Número de columnas
    """
    if not imagenes:
        print(f"No hay imágenes para visualizar: {titulo}")
        return
    
    n = min(filas * columnas, len(imagenes))
    fig, axes = plt.subplots(filas, columnas, figsize=(4*columnas, 3*filas))
    
    # Si solo hay una imagen, axes no es un array
    if n == 1:
        axes = np.array([axes])
    
    axes = axes.flatten()
    
    for i in range(n):
        img = imagenes[i]
        if img is None:
            axes[i].set_title(f"#{i+1} None")
            axes[i].axis('off')
            continue
            
        # Convertir BGR a RGB si es necesario
        if len(img.shape) == 3 and img.shape[2] == 3:
            disp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cmap = None
        else:
            disp = img
            cmap = 'gray'
            
        axes[i].imshow(disp, cmap=cmap)
        axes[i].set_title(f"#{i+1} {disp.shape}")
        axes[i].axis('off')
    
    # Ocultar ejes sobrantes
    for i in range(n, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(titulo)
    plt.tight_layout()
    plt.show()


# Ejemplo de uso de las funciones definidas arriba

"""
testing_data = cargar_imagenes_secuenciales(
"data/testing_data/C-NMC_test_final_phase_data/", 
prefijo="", 
max_imagenes=500, 
inicio=1, 
fin=2587
)  """"""


 
visualizar_imagenes(testing_data, titulo="Testing Data Original", filas=2, columnas=3)



"""""""
training_data_fold_0 = cargar_training_original(
    "data/training_data/fold_0/all/",
    max_imagenes=500
)

visualizar_imagenes(training_data_fold_0, titulo="Training Data Original Fold 0", filas=2, columnas=3)
"""

"""""""""
# Testing reconstruida
testing_reconstruida = cargar_imagenes_secuenciales(
    "data_reconstruida/testing_data/",
    prefijo="testing_",
    max_imagenes=500,
    inicio=1,
    fin=2587
)
"""""

"""""
training_fold_0_all_reconstruida = cargar_imagenes_secuenciales(
    "data_reconstruida/training_data/fold_0/all/",
    prefijo="testing_fold_0_",
    max_imagenes=500,
    inicio=1,
    fin=2587
)

#visualizar_imagenes(testing_reconstruida, titulo="Testing Data Reconstruida", filas=2, columnas=3)

visualizar_imagenes(training_fold_0_all_reconstruida, titulo="Testing Data Reconstruida Fold 0 all", filas=2, columnas=3)
"""""""""