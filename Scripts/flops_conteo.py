import numpy as np
import time
import os
from Carga_imagenes import cargar_imagenes_secuenciales, cargar_training_all_original, cargar_training_hem_original
from pca_flops import aplicar_pca_flops

def cargar_todos_datasets_originales():
    datasets = {}
    
    print("Cargando datasets...")
    
    datasets['testing_data'] = cargar_imagenes_secuenciales(
        "data/testing_data/C-NMC_test_final_phase_data/", 
        max_imagenes=500
    )
    
    datasets['fold_0_all'] = cargar_training_all_original("data/training_data/fold_0/all/")
    datasets['fold_0_hem'] = cargar_training_hem_original("data/training_data/fold_0/hem/")
    datasets['fold_1_all'] = cargar_training_all_original("data/training_data/fold_1/all/")
    datasets['fold_1_hem'] = cargar_training_hem_original("data/training_data/fold_1/hem/")
    datasets['fold_2_all'] = cargar_training_all_original("data/training_data/fold_2/all/")
    datasets['fold_2_hem'] = cargar_training_hem_original("data/training_data/fold_2/hem/")
    
    datasets['validation_data'] = cargar_imagenes_secuenciales(
        "data/validation_data/C-NMC_test_prelim_phase_data/",
        max_imagenes=500
    )
    return datasets

def guardar_resultados_flops(resultados):
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(logs_dir, f"flops_analysis_{timestamp}.txt")
    
    with open(filepath, 'w') as f:
        f.write("RESULTADOS FLOPS\n")
        f.write("=" * 50 + "\n")
        
        for nombre, datos in resultados.items():
            flops = datos['flops_data']['total_flops']
            tiempo = datos['flops_data']['total_time']
            f.write(f"{nombre}: {flops:,} FLOPS, {tiempo:.2f}s\n")
        
        f.write("\n")
        f.write("DETALLE POR OPERACION:\n")
        if resultados:
            primer_dato = next(iter(resultados.values()))
            for op, valor in primer_dato['flops_data'].items():
                if op not in ['total_time', 'total_flops']:
                    f.write(f"  {op}: {valor:,}\n")
    
    print(f"Log guardado: {filepath}")

def main():
    print("CONTEO DE FLOPS PARA PCA")
    print("=" * 50)
    
    datasets = cargar_todos_datasets_originales()
    resultados = {}
    
    for nombre, imagenes in datasets.items():
        if imagenes and len(imagenes) >= 500:
            print(f"Procesando {nombre}...")
            resultado = aplicar_pca_flops(imagenes)
            if resultado:
                imagenes_reconstruidas, imagenes_gray, shape, flops_data = resultado
                resultados[nombre] = {
                    'flops_data': flops_data,
                    'imagenes_procesadas': len(imagenes_reconstruidas)
                }
                print(f"  Completado: {flops_data['total_flops']:,} FLOPS")
    
    if resultados:
        guardar_resultados_flops(resultados)
        print("Analisis completado")
    else:
        print("No se pudieron procesar los datasets")

if __name__ == "__main__":
    main()