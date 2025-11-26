from Carga_imagenes import cargar_training_all_original, cargar_training_hem_original

def cargar_todos_datasets_con_labels():
    datasets = {}
    labels = {}
    
    print("Cargando datasets con labels...")
    
    # Fold 0
    datasets['fold_0_all'] = cargar_training_all_original("data/training_data/fold_0/all/")
    datasets['fold_0_hem'] = cargar_training_hem_original("data/training_data/fold_0/hem/")
    labels['fold_0_all'] = [0] * len(datasets['fold_0_all'])  # 0 = Healthy
    labels['fold_0_hem'] = [1] * len(datasets['fold_0_hem'])  # 1 = Leukemia
    
    # Fold 1
    datasets['fold_1_all'] = cargar_training_all_original("data/training_data/fold_1/all/")
    datasets['fold_1_hem'] = cargar_training_hem_original("data/training_data/fold_1/hem/")
    labels['fold_1_all'] = [0] * len(datasets['fold_1_all'])  # 0 = Healthy
    labels['fold_1_hem'] = [1] * len(datasets['fold_1_hem'])  # 1 = Leukemia
    
    # Fold 2
    datasets['fold_2_all'] = cargar_training_all_original("data/training_data/fold_2/all/")
    datasets['fold_2_hem'] = cargar_training_hem_original("data/training_data/fold_2/hem/")
    labels['fold_2_all'] = [0] * len(datasets['fold_2_all'])  # 0 = Healthy
    labels['fold_2_hem'] = [1] * len(datasets['fold_2_hem'])  # 1 = Leukemia
    
    # Resumen
    print("\nResumen de datasets cargados:")
    for key in datasets:
        print(f"  {key}: {len(datasets[key])} imágenes, labels: {len(labels[key])}")
    
    return datasets, labels


if __name__ == "__main__":
    datasets, labels = cargar_todos_datasets_con_labels()

    # Ejemplo de acceso a un dataset y sus labels
    fold_0_all_data = datasets['fold_0_all']
    fold_0_all_labels = labels['fold_0_all']
    print(f"\nEjemplo - Fold 0 All: {len(fold_0_all_data)} imágenes, {len(fold_0_all_labels)} labels")