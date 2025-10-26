# 🎯 Пошаговый план адаптации YOLOv10 + DINOv2

**Ваши данные:** 10,000 размеченных + 300,000 неразмеченных изображений

---

## 📋 **ЭТАП 0: Подготовка и анализ данных (1-2 дня)**

### Шаг 0.1: Установка зависимостей
```bash
# Основные библиотеки
pip install torch torchvision ultralytics
pip install transformers timm
pip install opencv-python albumentations
pip install scikit-learn matplotlib seaborn

# Для DINOv2
pip install git+https://github.com/facebookresearch/dinov2.git
```

### Шаг 0.2: Анализ данных
```python
import os
import cv2
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

def analyze_dataset(image_folder, annotation_folder=None):
    """Анализ вашего датасета"""
    
    # Базовая статистика изображений
    image_stats = {
        'total_images': 0,
        'resolutions': [],
        'aspect_ratios': [],
        'file_sizes': []
    }
    
    for img_file in os.listdir(image_folder):
        if img_file.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(image_folder, img_file)
            img = cv2.imread(img_path)
            
            if img is not None:
                h, w = img.shape[:2]
                image_stats['total_images'] += 1
                image_stats['resolutions'].append((w, h))
                image_stats['aspect_ratios'].append(w/h)
                image_stats['file_sizes'].append(os.path.getsize(img_path))
    
    # Анализ разметки (если есть)
    if annotation_folder:
        class_distribution = analyze_annotations(annotation_folder)
        return image_stats, class_distribution
    
    return image_stats, None

def analyze_annotations(annotation_folder):
    """Анализ распределения классов"""
    class_counts = Counter()
    bbox_sizes = []
    
    for ann_file in os.listdir(annotation_folder):
        if ann_file.endswith('.txt'):  # YOLO формат
            with open(os.path.join(annotation_folder, ann_file), 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        w, h = float(parts[3]), float(parts[4])
                        
                        class_counts[class_id] += 1
                        bbox_sizes.append(w * h)
    
    return {
        'class_distribution': dict(class_counts),
        'bbox_sizes': bbox_sizes,
        'total_objects': sum(class_counts.values())
    }

# Выполните анализ ваших данных
labeled_stats, class_info = analyze_dataset(
    "path/to/labeled/images", 
    "path/to/annotations"
)
unlabeled_stats, _ = analyze_dataset("path/to/unlabeled/images")

print(f"Размеченных изображений: {labeled_stats['total_images']}")
print(f"Неразмеченных изображений: {unlabeled_stats['total_images']}")
print(f"Распределение классов: {class_info['class_distribution']}")
```

---

## 📊 **ЭТАП 1: Baseline обучение YOLOv10 (2-3 дня)**

### Шаг 1.1: Подготовка данных для YOLOv10
```python
import yaml
from sklearn.model_selection import train_test_split

def prepare_yolo_dataset(image_folder, annotation_folder, output_folder):
    """Подготовка датасета в формате YOLOv10"""
    
    # Создание структуры папок
    os.makedirs(f"{output_folder}/images/train", exist_ok=True)
    os.makedirs(f"{output_folder}/images/val", exist_ok=True)
    os.makedirs(f"{output_folder}/labels/train", exist_ok=True)
    os.makedirs(f"{output_folder}/labels/val", exist_ok=True)
    
    # Получение списка файлов
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]
    
    # Разделение на train/val (80/20)
    train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)
    
    # Копирование файлов
    import shutil
    
    for split, files in [('train', train_files), ('val', val_files)]:
        for img_file in files:
            # Изображение
            src_img = os.path.join(image_folder, img_file)
            dst_img = os.path.join(output_folder, f"images/{split}", img_file)
            shutil.copy2(src_img, dst_img)
            
            # Аннотация
            ann_file = img_file.replace('.jpg', '.txt').replace('.png', '.txt')
            src_ann = os.path.join(annotation_folder, ann_file)
            dst_ann = os.path.join(output_folder, f"labels/{split}", ann_file)
            if os.path.exists(src_ann):
                shutil.copy2(src_ann, dst_ann)
    
    # Создание конфига
    config = {
        'train': f"{output_folder}/images/train",
        'val': f"{output_folder}/images/val",
        'nc': len(class_info['class_distribution']),  # количество классов
        'names': list(range(len(class_info['class_distribution'])))  # замените на реальные имена
    }
    
    with open(f"{output_folder}/dataset.yaml", 'w') as f:
        yaml.dump(config, f)
    
    return f"{output_folder}/dataset.yaml"

# Подготовка датасета
dataset_config = prepare_yolo_dataset(
    "path/to/labeled/images",
    "path/to/annotations", 
    "prepared_dataset"
)
```

### Шаг 1.2: Baseline обучение YOLOv10
```python
from ultralytics import YOLO

def train_baseline_yolo():
    """Обучение baseline YOLOv10"""
    
    # Загрузка предобученной модели
    model = YOLO('yolov10s.pt')  # или yolov10m.pt для лучшей точности
    
    # Обучение
    results = model.train(
        data=dataset_config,
        epochs=100,
        imgsz=640,
        batch=16,  # настройте под вашу GPU
        device=0,  # GPU
        name='baseline_yolov10',
        patience=20,  # early stopping
        save_period=10,  # сохранять каждые 10 эпох
        
        # Аугментации
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
    )
    
    # Валидация
    val_results = model.val()
    
    print(f"Baseline mAP@0.5: {val_results.box.map50:.3f}")
    print(f"Baseline mAP@0.5:0.95: {val_results.box.map:.3f}")
    
    return model, val_results

# Обучение baseline модели
baseline_model, baseline_metrics = train_baseline_yolo()
```

---

## 🔧 **ЭТАП 2: Извлечение признаков DINOv2 (1 день)**

### Шаг 2.1: Извлечение признаков для всех изображений
```python
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    """Датасет для извлечения признаков"""
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.image_files = [f for f in os.listdir(image_folder) 
                           if f.endswith(('.jpg', '.png'))]
        
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, self.image_files[idx]

def extract_dino_features(image_folder, output_file, batch_size=32):
    """Извлечение признаков DINOv2 для всех изображений"""
    
    # Загрузка DINOv2
    dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    dino_model.eval()
    dino_model = dino_model.cuda()
    
    # Датасет и даталоадер
    dataset = ImageDataset(image_folder)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    features_dict = {}
    
    print(f"Извлечение признаков для {len(dataset)} изображений...")
    
    with torch.no_grad():
        for batch_idx, (images, filenames) in enumerate(dataloader):
            images = images.cuda()
            
            # Извлечение признаков
            features = dino_model(images)  # [batch_size, 768]
            
            # Сохранение в словарь
            for feat, filename in zip(features, filenames):
                features_dict[filename] = feat.cpu().numpy()
            
            if batch_idx % 100 == 0:
                print(f"Обработано батчей: {batch_idx}/{len(dataloader)}")
    
    # Сохранение признаков
    np.save(output_file, features_dict)
    print(f"Признаки сохранены в {output_file}")
    
    return features_dict

# Извлечение признаков для размеченных данных
labeled_features = extract_dino_features(
    "path/to/labeled/images", 
    "labeled_dino_features.npy"
)

# Извлечение признаков для неразмеченных данных (может занять время!)
unlabeled_features = extract_dino_features(
    "path/to/unlabeled/images", 
    "unlabeled_dino_features.npy"
)
```

### Шаг 2.2: Анализ признаков домена
```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def analyze_domain_gap(labeled_features, unlabeled_features, sample_size=5000):
    """Анализ различий между размеченными и неразмеченными данными"""
    
    # Случайная выборка для визуализации
    labeled_sample = np.random.choice(
        list(labeled_features.values()), 
        min(sample_size//2, len(labeled_features)), 
        replace=False
    )
    unlabeled_sample = np.random.choice(
        list(unlabeled_features.values()), 
        min(sample_size//2, len(unlabeled_features)), 
        replace=False
    )
    
    # Объединение данных
    all_features = np.vstack([
        np.stack(labeled_sample),
        np.stack(unlabeled_sample)
    ])
    
    labels = np.array(['labeled'] * len(labeled_sample) + 
                     ['unlabeled'] * len(unlabeled_sample))
    
    # PCA визуализация
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(all_features)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for label in ['labeled', 'unlabeled']:
        mask = labels == label
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                   alpha=0.6, label=label)
    plt.title('PCA проекция признаков DINOv2')
    plt.legend()
    
    # t-SNE визуализация
    tsne = TSNE(n_components=2, random_state=42)
    features_tsne = tsne.fit_transform(all_features)
    
    plt.subplot(1, 2, 2)
    for label in ['labeled', 'unlabeled']:
        mask = labels == label
        plt.scatter(features_tsne[mask, 0], features_tsne[mask, 1], 
                   alpha=0.6, label=label)
    plt.title('t-SNE проекция признаков DINOv2')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('domain_analysis.png', dpi=150)
    plt.show()
    
    # Вычисление domain gap
    from scipy.spatial.distance import cdist
    
    labeled_centroid = np.mean(np.stack(labeled_sample), axis=0)
    unlabeled_centroid = np.mean(np.stack(unlabeled_sample), axis=0)
    
    domain_gap = np.linalg.norm(labeled_centroid - unlabeled_centroid)
    
    print(f"Domain gap (Euclidean distance): {domain_gap:.3f}")
    
    return domain_gap

# Анализ различий между доменами
domain_gap = analyze_domain_gap(labeled_features, unlabeled_features)
```

---

## 🎯 **ЭТАП 3: Псевдо-разметка с DINOv2 (2-3 дня)**

### Шаг 3.1: Генерация начальных псевдо-меток
```python
def generate_initial_pseudo_labels(baseline_model, unlabeled_folder, 
                                 dino_features, confidence_threshold=0.7):
    """Генерация начальных псевдо-меток с высокой уверенностью"""
    
    pseudo_labels = {}
    high_confidence_files = []
    
    # Предсказания на неразмеченных данных
    results = baseline_model.predict(
        source=unlabeled_folder,
        conf=confidence_threshold,
        save=False,
        verbose=False
    )
    
    for result in results:
        filename = os.path.basename(result.path)
        boxes = result.boxes
        
        if boxes is not None and len(boxes) > 0:
            # Фильтрация через семантическую согласованность DINOv2
            valid_detections = []
            
            for box in boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                xyxy = box.xyxy[0].cpu().numpy()
                
                # Дополнительная проверка через DINOv2 (опционально)
                semantic_score = check_semantic_consistency(
                    result.orig_img, xyxy, cls, dino_features[filename]
                )
                
                if semantic_score > 0.6:  # порог семантической согласованности
                    valid_detections.append({
                        'bbox': xyxy,
                        'class': cls,
                        'confidence': conf * semantic_score  # взвешенная уверенность
                    })
            
            if valid_detections:
                pseudo_labels[filename] = valid_detections
                high_confidence_files.append(filename)
    
    print(f"Сгенерировано псевдо-меток для {len(pseudo_labels)} изображений")
    return pseudo_labels, high_confidence_files

def check_semantic_consistency(image, bbox, predicted_class, dino_features):
    """Проверка семантической согласованности через DINOv2"""
    
    # Извлечение crop'а объекта
    x1, y1, x2, y2 = bbox.astype(int)
    crop = image[y1:y2, x1:x2]
    
    if crop.size == 0:
        return 0.0
    
    # Получение DINOv2 признаков crop'а
    crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    crop_tensor = transform(crop_pil).unsqueeze(0).cuda()
    
    with torch.no_grad():
        dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').cuda()
        crop_features = dino_model(crop_tensor)[0].cpu().numpy()
    
    # Сравнение с глобальными признаками изображения
    similarity = np.dot(crop_features, dino_features) / (
        np.linalg.norm(crop_features) * np.linalg.norm(dino_features)
    )
    
    return max(0.0, similarity)

# Генерация начальных псевдо-меток
initial_pseudo_labels, confident_files = generate_initial_pseudo_labels(
    baseline_model, 
    "path/to/unlabeled/images",
    unlabeled_features,
    confidence_threshold=0.8
)
```

### Шаг 3.2: Итеративное улучшение псевдо-меток
```python
def iterative_pseudo_labeling(model, unlabeled_data, labeled_data, 
                             dino_features, num_iterations=5):
    """Итеративное улучшение псевдо-меток"""
    
    current_model = model
    iteration_results = []
    
    for iteration in range(num_iterations):
        print(f"\n=== Итерация {iteration + 1} ===")
        
        # 1. Генерация псевдо-меток текущей моделью
        confidence_thresh = 0.9 - iteration * 0.1  # постепенное снижение порога
        
        pseudo_labels, confident_files = generate_initial_pseudo_labels(
            current_model,
            unlabeled_data,
            dino_features,
            confidence_threshold=confidence_thresh
        )
        
        print(f"Псевдо-меток: {len(pseudo_labels)}")
        
        # 2. Отбор лучших псевдо-меток
        best_pseudo_labels = select_best_pseudo_labels(
            pseudo_labels, dino_features, top_k=min(5000, len(pseudo_labels))
        )
        
        print(f"Отобрано лучших: {len(best_pseudo_labels)}")
        
        # 3. Создание расширенного датасета
        extended_dataset = create_extended_dataset(
            labeled_data, best_pseudo_labels
        )
        
        # 4. Переобучение модели
        current_model = retrain_model(current_model, extended_dataset)
        
        # 5. Валидация
        val_results = current_model.val()
        iteration_results.append({
            'iteration': iteration + 1,
            'pseudo_labels_count': len(best_pseudo_labels),
            'mAP_50': val_results.box.map50,
            'mAP_50_95': val_results.box.map
        })
        
        print(f"mAP@0.5: {val_results.box.map50:.3f}")
        print(f"mAP@0.5:0.95: {val_results.box.map:.3f}")
    
    return current_model, iteration_results

def select_best_pseudo_labels(pseudo_labels, dino_features, top_k=5000):
    """Отбор лучших псевдо-меток на основе уверенности и согласованности"""
    
    scored_labels = []
    
    for filename, detections in pseudo_labels.items():
        for det in detections:
            # Комбинированный скор: confidence + semantic consistency
            combined_score = det['confidence']
            
            scored_labels.append({
                'filename': filename,
                'detection': det,
                'score': combined_score
            })
    
    # Сортировка по убыванию скора
    scored_labels.sort(key=lambda x: x['score'], reverse=True)
    
    # Отбор top-k
    best_labels = {}
    for item in scored_labels[:top_k]:
        filename = item['filename']
        if filename not in best_labels:
            best_labels[filename] = []
        best_labels[filename].append(item['detection'])
    
    return best_labels

def create_extended_dataset(labeled_data_path, pseudo_labels):
    """Создание расширенного датасета с псевдо-метками"""
    
    extended_path = "extended_dataset"
    os.makedirs(f"{extended_path}/images/train", exist_ok=True)
    os.makedirs(f"{extended_path}/labels/train", exist_ok=True)
    
    # Копирование исходных размеченных данных
    import shutil
    shutil.copytree(f"{labeled_data_path}/images", f"{extended_path}/images", dirs_exist_ok=True)
    shutil.copytree(f"{labeled_data_path}/labels", f"{extended_path}/labels", dirs_exist_ok=True)
    
    # Добавление псевдо-размеченных данных
    for filename, detections in pseudo_labels.items():
        # Копирование изображения
        src_img = f"path/to/unlabeled/images/{filename}"
        dst_img = f"{extended_path}/images/train/{filename}"
        shutil.copy2(src_img, dst_img)
        
        # Создание YOLO аннотации
        ann_filename = filename.replace('.jpg', '.txt').replace('.png', '.txt')
        with open(f"{extended_path}/labels/train/{ann_filename}", 'w') as f:
            for det in detections:
                # Конвертация в YOLO формат
                x1, y1, x2, y2 = det['bbox']
                img = cv2.imread(src_img)
                h, w = img.shape[:2]
                
                x_center = (x1 + x2) / 2 / w
                y_center = (y1 + y2) / 2 / h
                width = (x2 - x1) / w
                height = (y2 - y1) / h
                
                f.write(f"{det['class']} {x_center} {y_center} {width} {height}\n")
    
    # Обновление конфига
    config = {
        'train': f"{extended_path}/images/train",
        'val': f"{labeled_data_path}/images/val",  # валидация на исходных данных
        'nc': len(class_info['class_distribution']),
        'names': list(range(len(class_info['class_distribution'])))
    }
    
    with open(f"{extended_path}/dataset.yaml", 'w') as f:
        yaml.dump(config, f)
    
    return f"{extended_path}/dataset.yaml"

def retrain_model(model, dataset_config):
    """Переобучение модели на расширенном датасете"""
    
    # Дообучение с меньшим learning rate
    results = model.train(
        data=dataset_config,
        epochs=30,  # меньше эпох для итеративного обучения
        imgsz=640,
        batch=16,
        device=0,
        name=f'pseudo_iteration_{hash(dataset_config)}',
        resume=False,
        
        # Меньший learning rate для стабильности
        lr0=0.001,
        patience=10,
        
        # Аугментации
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.3,
        fliplr=0.5,
        mosaic=0.5,  # уменьшенная мозаика для псевдо-меток
    )
    
    return model

# Запуск итеративного псевдо-разметки
improved_model, iteration_history = iterative_pseudo_labeling(
    baseline_model,
    "path/to/unlabeled/images",
    "prepared_dataset",
    unlabeled_features,
    num_iterations=3  # начните с 3 итераций
)
```

---

## 🔧 **ЭТАП 4: Knowledge Distillation YOLOv10 ← DINOv2 (3-4 дня)**

### Шаг 4.1: Реализация дистилляции
```python
import torch.nn as nn
import torch.nn.functional as F

class YOLODINODistillation(nn.Module):
    """Модуль для дистилляции знаний DINOv2 → YOLOv10"""
    
    def __init__(self, yolo_model, dino_model, distillation_layers=['backbone']):
        super().__init__()
        
        self.yolo_model = yolo_model.model  # получаем внутреннюю модель
        self.dino_model = dino_model
        self.distillation_layers = distillation_layers
        
        # Проекционные слои для согласования размерностей
        self.projectors = nn.ModuleDict()
        
        # Для backbone уровней YOLOv10
        backbone_dims = [256, 512, 1024]  # примерные размерности для YOLOv10s
        dino_dim = 768  # для DINOv2-B
        
        for i, dim in enumerate(backbone_dims):
            self.projectors[f'level_{i}'] = nn.Sequential(
                nn.Conv2d(dim, dino_dim, 1),
                nn.BatchNorm2d(dino_dim),
                nn.ReLU(),
                nn.Conv2d(dino_dim, dino_dim, 3, padding=1),
                nn.BatchNorm2d(dino_dim),
                nn.AdaptiveAvgPool2d(1)  # глобальный пулинг
            )
    
    def extract_yolo_features(self, x):
        """Извлечение промежуточных признаков YOLOv10"""
        features = []
        
        # Проход через backbone
        for i, layer in enumerate(self.yolo_model[:10]):  # backbone layers
            x = layer(x)
            if i in [4, 6, 8]:  # примерные слои для извлечения
                features.append(x)
        
        return features
    
    def forward(self, images, targets=None):
        """Прямой проход с дистилляцией"""
        
        # Стандартные предсказания YOLOv10
        yolo_outputs = self.yolo_model(images)
        
        # Извлечение промежуточных признаков YOLOv10
        yolo_features = self.extract_yolo_features(images)
        
        # Извлечение признаков DINOv2
        with torch.no_grad():
            dino_features = self.dino_model(F.interpolate(images, size=224))
        
        # Вычисление дистилляционных потерь
        distillation_losses = {}
        
        for i, yolo_feat in enumerate(yolo_features):
            # Проекция YOLOv10 признаков
            projected = self.projectors[f'level_{i}'](yolo_feat)
            projected = projected.view(projected.size(0), -1)  # flatten
            
            # Нормализация для косинусного расстояния
            projected_norm = F.normalize(projected, p=2, dim=1)
            dino_norm = F.normalize(dino_features, p=2, dim=1)
            
            # Дистилляционная потеря
            distill_loss = 1 - F.cosine_similarity(projected_norm, dino_norm).mean()
            distillation_losses[f'distill_level_{i}'] = distill_loss
        
        if self.training and targets is not None:
            return yolo_outputs, distillation_losses
        else:
            return yolo_outputs

def train_with_distillation(yolo_model, train_dataloader, val_dataloader, 
                           dino_features_dict, num_epochs=50):
    """Обучение YOLOv10 с дистилляцией DINOv2"""
    
    # Инициализация DINOv2
    dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    dino_model.eval()
    dino_model.cuda()
    
    # Создание дистилляционной модели
    distill_model = YOLODINODistillation(yolo_model, dino_model)
    distill_model.cuda()
    distill_model.train()
    
    # Оптимизатор
    optimizer = torch.optim.AdamW(distill_model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_map = 0.0
    distillation_weight = 0.5  # вес дистилляционной потери
    
    for epoch in range(num_epochs):
        distill_model.train()
        epoch_losses = {'detection': [], 'distillation': []}
        
        for batch_idx, (images, targets) in enumerate(train_dataloader):
            images = images.cuda()
            
            # Прямой проход
            yolo_outputs, distill_losses = distill_model(images, targets)
            
            # Детекционная потеря (стандартная YOLO потеря)
            detection_loss = compute_yolo_loss(yolo_outputs, targets)
            
            # Общая дистилляционная потеря
            total_distill_loss = sum(distill_losses.values()) / len(distill_losses)
            
            # Общая потеря
            total_loss = detection_loss + distillation_weight * total_distill_loss
            
            # Обратное распространение
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            epoch_losses['detection'].append(detection_loss.item())
            epoch_losses['distillation'].append(total_distill_loss.item())
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}: "
                      f"Det Loss: {detection_loss:.4f}, "
                      f"Distill Loss: {total_distill_loss:.4f}")
        
        # Валидация
        val_map = validate_model(distill_model, val_dataloader)
        
        print(f"Epoch {epoch+1}: "
              f"Avg Det Loss: {np.mean(epoch_losses['detection']):.4f}, "
              f"Avg Distill Loss: {np.mean(epoch_losses['distillation']):.4f}, "
              f"Val mAP: {val_map:.4f}")
        
        # Сохранение лучшей модели
        if val_map > best_map:
            best_map = val_map
            torch.save(distill_model.state_dict(), 'best_distilled_model.pth')
        
        scheduler.step()
        
        # Адаптивное изменение веса дистилляции
        if epoch > num_epochs // 2:
            distillation_weight *= 0.99  # постепенное уменьшение
    
    return distill_model, best_map

# Запуск дистилляции (упрощенная версия)
# В реальности нужно адаптировать под ваш конкретный dataloader
print("Начинаем knowledge distillation...")
print("Примечание: Полная реализация требует интеграции с вашим training pipeline")
```

### Шаг 4.2: Альтернативный подход - Feature-level дистилляция
```python
def simpler_feature_distillation(improved_model, labeled_dataloader, 
                                unlabeled_dataloader, num_epochs=20):
    """Упрощенный подход к дистилляции на уровне признаков"""
    
    # Загрузка DINOv2
    dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    dino_model.eval().cuda()
    
    # Создание проекционного слоя
    projector = nn.Sequential(
        nn.Linear(1000, 768),  # от YOLOv10 к DINOv2 размерности
        nn.ReLU(),
        nn.Linear(768, 768)
    ).cuda()
    
    # Оптимизатор только для проекционного слоя
    optimizer = torch.optim.Adam(
        list(improved_model.parameters()) + list(projector.parameters()),
        lr=1e-4
    )
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        # Обучение на неразмеченных данных с дистилляцией
        for images in unlabeled_dataloader:
            images = images.cuda()
            
            # YOLOv10 признаки (используем предпоследний слой)
            with torch.no_grad():
                yolo_features = improved_model.model[-2](
                    improved_model.model[:-2](images)
                ).mean(dim=[2, 3])  # global average pooling
            
            # DINOv2 признаки
            with torch.no_grad():
                resized_images = F.interpolate(images, size=224)
                dino_features = dino_model(resized_images)
            
            # Проекция YOLOv10 признаков
            projected_yolo = projector(yolo_features)
            
            # Дистилляционная потеря
            distill_loss = F.mse_loss(projected_yolo, dino_features.detach())
            
            optimizer.zero_grad()
            distill_loss.backward()
            optimizer.step()
            
            total_loss += distill_loss.item()
        
        print(f"Epoch {epoch+1}: Distillation Loss = {total_loss/len(unlabeled_dataloader):.4f}")
    
    return improved_model

# Это более простая альтернатива, если полная дистилляция слишком сложна
```

---

## 📊 **ЭТАП 5: Финальная оценка и оптимизация (1-2 дня)**

### Шаг 5.1: Комплексная оценка
```python
def comprehensive_evaluation(models_dict, test_dataloader):
    """Комплексная оценка всех моделей"""
    
    results = {}
    
    for model_name, model in models_dict.items():
        print(f"\nОценка модели: {model_name}")
        
        # Стандартные метрики детекции
        val_results = model.val()
        
        # Дополнительные метрики
        additional_metrics = compute_additional_metrics(model, test_dataloader)
        
        results[model_name] = {
            'mAP_50': val_results.box.map50,
            'mAP_50_95': val_results.box.map,
            'precision': val_results.box.p,
            'recall': val_results.box.r,
            **additional_metrics
        }
    
    # Создание сводной таблицы
    import pandas as pd
    
    df_results = pd.DataFrame(results).T
    print("\n=== СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ ===")
    print(df_results.round(4))
    
    # Сохранение результатов
    df_results.to_csv('model_comparison_results.csv')
    
    # Визуализация
    plot_results_comparison(df_results)
    
    return results

def compute_additional_metrics(model, dataloader):
    """Дополнительные метрики оценки"""
    
    all_predictions = []
    inference_times = []
    
    model.eval()
    with torch.no_grad():
        for images, targets in dataloader:
            start_time = time.time()
            
            predictions = model(images)
            
            inference_time = (time.time() - start_time) / len(images)
            inference_times.append(inference_time)
            
            all_predictions.extend(predictions)
    
    return {
        'avg_inference_time_ms': np.mean(inference_times) * 1000,
        'std_inference_time_ms': np.std(inference_times) * 1000,
        'total_predictions': len(all_predictions)
    }

def plot_results_comparison(df_results):
    """Визуализация сравнения результатов"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # mAP@0.5
    axes[0, 0].bar(df_results.index, df_results['mAP_50'])
    axes[0, 0].set_title('mAP@0.5')
    axes[0, 0].set_ylabel('mAP')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # mAP@0.5:0.95
    axes[0, 1].bar(df_results.index, df_results['mAP_50_95'])
    axes[0, 1].set_title('mAP@0.5:0.95')
    axes[0, 1].set_ylabel('mAP')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Precision vs Recall
    axes[1, 0].scatter(df_results['recall'], df_results['precision'], s=100)
    for i, model in enumerate(df_results.index):
        axes[1, 0].annotate(model, (df_results['recall'].iloc[i], 
                                   df_results['precision'].iloc[i]))
    axes[1, 0].set_xlabel('Recall')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title('Precision vs Recall')
    
    # Время инференса
    axes[1, 1].bar(df_results.index, df_results['avg_inference_time_ms'])
    axes[1, 1].set_title('Среднее время инференса (мс)')
    axes[1, 1].set_ylabel('Время (мс)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

# Итоговая оценка всех моделей
models_to_evaluate = {
    'baseline_yolov10': baseline_model,
    'pseudo_labeled': improved_model,
    # 'distilled': distilled_model,  # если реализована дистилляция
}

final_results = comprehensive_evaluation(models_to_evaluate, test_dataloader)
```

### Шаг 5.2: Выбор и экспорт лучшей модели
```python
def select_and_export_best_model(results_dict, models_dict, 
                                main_metric='mAP_50', 
                                secondary_metric='mAP_50_95'):
    """Выбор и экспорт лучшей модели"""
    
    # Выбор лучшей модели
    best_model_name = max(results_dict.keys(), 
                         key=lambda x: results_dict[x][main_metric])
    
    best_model = models_dict[best_model_name]
    best_score = results_dict[best_model_name][main_metric]
    
    print(f"\nЛучшая модель: {best_model_name}")
    print(f"{main_metric}: {best_score:.4f}")
    
    # Экспорт в разные форматы
    export_formats = ['onnx', 'torchscript', 'tflite']
    
    for format_name in export_formats:
        try:
            export_path = f"best_model_{best_model_name}.{format_name}"
            
            if format_name == 'onnx':
                best_model.export(format='onnx', optimize=True)
            elif format_name == 'torchscript':
                best_model.export(format='torchscript')
            elif format_name == 'tflite':
                best_model.export(format='tflite')
            
            print(f"Модель экспортирована в {format_name}: {export_path}")
            
        except Exception as e:
            print(f"Ошибка экспорта в {format_name}: {e}")
    
    # Сохранение конфигурации и метрик
    model_config = {
        'best_model_name': best_model_name,
        'metrics': results_dict[best_model_name],
        'training_details': {
            'labeled_samples': 10000,
            'unlabeled_samples': 300000,
            'training_strategy': 'YOLOv10 + DINOv2 + Pseudo-labeling',
            'final_dataset_size': 'labeled + selected_pseudo_labeled'
        }
    }
    
    import json
    with open('best_model_config.json', 'w') as f:
        json.dump(model_config, f, indent=2)
    
    return best_model, best_model_name

# Выбор и экспорт лучшей модели
best_model, best_name = select_and_export_best_model(
    final_results, models_to_evaluate
)
```

---

## 📋 **РЕЗЮМЕ И ВРЕМЕННЫЕ ЗАТРАТЫ**

### Общий timeline (8-12 дней):
```
День 1-2:   Анализ данных + подготовка
День 3-4:   Baseline YOLOv10 обучение  
День 5:     Извлечение DINOv2 признаков
День 6-8:   Псевдо-разметка (итеративная)
День 9-11:  Knowledge Distillation (опционально)
День 12:    Финальная оценка + экспорт
```

### Ожидаемые улучшения:
- **Baseline YOLOv10**: mAP@0.5 ~ 0.65-0.75
- **+ Псевдо-разметка**: +3-7% mAP
- **+ Дистилляция**: +1-3% mAP

### Ключевые файлы для сохранения:
```
best_model_[name].pt          # Лучшая модель
best_model_config.json        # Конфигурация
model_comparison_results.csv  # Сравнение всех моделей
labeled_dino_features.npy     # DINOv2 признаки (размеченные)
unlabeled_dino_features.npy   # DINOv2 признаки (неразмеченные)
```

### Практические советы:
1. **Начните с простого**: сначала baseline, затем псевдо-разметка
2. **Мониторьте качество**: регулярно проверяйте на валидационной выборке
3. **Экономьте время**: дистилляция опциональна, основной прирост от псевдо-разметки
4. **Сохраняйте промежуточные результаты**: каждый этап может пригодиться
5. **Адаптируйте под ресурсы**: batch_size и количество эпох под вашу GPU

**Готов помочь с реализацией любого из этапов! 🚀**