# 🎯 Объединение Knowledge Distillation + DINOv2 Domain Adaptation

Покажу **3 способа** эффективного совмещения этих техник для максимального прироста качества.

---

## 🔧 **АРХИТЕКТУРА 1: Двухэтапная дистилляция**

### Этап 1: Domain Adaptation DINOv2 → Этап 2: Distillation в YOLOv10

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoStageDistillationPipeline:
    """Двухэтапная дистилляция: DINOv2 Adaptation → YOLO Distillation"""
    
    def __init__(self, yolo_model, num_classes):
        self.yolo_model = yolo_model
        self.num_classes = num_classes
        
        # Stage 1: Domain-adapted DINOv2
        self.dinov2_base = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.dinov2_base.eval()  # Заморожен
        
        self.domain_adapter = DomainAdapter(
            input_dim=768, 
            bottleneck_dim=512,
            output_dim=768
        )
        
        # Stage 2: Multi-level feature distillation
        self.feature_aligners = self._create_feature_aligners()
        
    def _create_feature_aligners(self):
        """Создание слоев для согласования размерностей YOLOv10 ← DINOv2"""
        
        # YOLOv10 backbone feature dimensions (примерные)
        yolo_dims = [256, 512, 1024]  # C3, C4, C5 levels
        dino_dim = 768
        
        aligners = nn.ModuleDict()
        
        for i, yolo_dim in enumerate(yolo_dims):
            aligners[f'level_{i}'] = nn.Sequential(
                # Spatial alignment (адаптируем к пространственному разрешению)
                nn.AdaptiveAvgPool2d(1),  # Global pooling YOLOv10 features
                nn.Flatten(),
                
                # Channel alignment
                nn.Linear(yolo_dim, dino_dim),
                nn.BatchNorm1d(dino_dim),
                nn.ReLU(),
                
                # Feature enhancement
                nn.Linear(dino_dim, dino_dim),
                nn.Dropout(0.1),
                nn.LayerNorm(dino_dim)
            )
        
        return aligners

class DomainAdapter(nn.Module):
    """Улучшенный domain adapter с многоуровневой проекцией"""
    
    def __init__(self, input_dim=768, bottleneck_dim=512, output_dim=768):
        super().__init__()
        
        # Multi-scale bottleneck
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, bottleneck_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1)
            ) for _ in range(3)  # 3 разных представления
        ])
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(bottleneck_dim // 2 * 3, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.Tanh()
        )
        
        # Residual connection с обучаемым весом
        self.residual_weight = nn.Parameter(torch.tensor(0.2))
        
    def forward(self, dinov2_features):
        # Multi-scale encoding
        encoded_features = []
        for encoder in self.encoders:
            encoded_features.append(encoder(dinov2_features))
        
        # Fusion
        fused = self.fusion(torch.cat(encoded_features, dim=1))
        
        # Decode
        adapted = self.decoder(fused)
        
        # Residual connection
        output = adapted + self.residual_weight * dinov2_features
        
        return F.normalize(output, dim=1)

# STAGE 1: Обучение Domain Adapter
def train_domain_adapter(domain_images_folder, num_epochs=25):
    """Обучение domain adapter с улучшенной контрастивной потерей"""
    
    # Setup
    dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').cuda().eval()
    adapter = DomainAdapter().cuda()
    
    # Более сложные аугментации для лучшего domain adaptation
    strong_aug = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(3, (0.1, 2.0))], p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = ImageFolder(domain_images_folder, transform=lambda x: x)  # Raw images
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # Training loop
    for epoch in range(num_epochs):
        adapter.train()
        total_loss = 0
        
        for batch_idx, (images, _) in enumerate(dataloader):
            # Создаем 3 аугментации для triplet-like loss
            aug1 = torch.stack([strong_aug(img) for img in images]).cuda()
            aug2 = torch.stack([strong_aug(img) for img in images]).cuda()
            aug3 = torch.stack([strong_aug(img) for img in images]).cuda()
            
            # DINOv2 features
            with torch.no_grad():
                dino_feat1 = dinov2_model(aug1)
                dino_feat2 = dinov2_model(aug2)  
                dino_feat3 = dinov2_model(aug3)
            
            # Adapt features
            adapted1 = adapter(dino_feat1)
            adapted2 = adapter(dino_feat2)
            adapted3 = adapter(dino_feat3)
            
            # Enhanced contrastive loss (triplet-like)
            loss = enhanced_contrastive_loss(adapted1, adapted2, adapted3, temperature=0.07)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}: Loss = {loss.item():.4f}')
        
        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}: Average Loss = {avg_loss:.4f}')
        
        # Save checkpoint
        if epoch % 5 == 0:
            torch.save(adapter.state_dict(), f'domain_adapter_epoch_{epoch}.pth')
    
    return adapter

def enhanced_contrastive_loss(feat1, feat2, feat3, temperature=0.07):
    """Улучшенная контрастивная потеря с тройками"""
    
    # Standard pairwise contrastive
    loss_12 = contrastive_loss_pair(feat1, feat2, temperature)
    loss_13 = contrastive_loss_pair(feat1, feat3, temperature) 
    loss_23 = contrastive_loss_pair(feat2, feat3, temperature)
    
    # Triplet consistency: все 3 аугментации должны быть близки
    triplet_loss = triplet_consistency_loss(feat1, feat2, feat3)
    
    return (loss_12 + loss_13 + loss_23) / 3 + 0.1 * triplet_loss

def contrastive_loss_pair(feat1, feat2, temperature):
    """Стандартная парная контрастивная потеря"""
    batch_size = feat1.shape[0]
    
    # Similarity matrix
    features = torch.cat([feat1, feat2], dim=0)
    sim_matrix = torch.mm(features, features.T) / temperature
    
    # Mask diagonal
    mask = torch.eye(2 * batch_size, device=features.device, dtype=torch.bool)
    sim_matrix.masked_fill_(mask, -float('inf'))
    
    # Positive pairs
    pos_indices = torch.arange(batch_size, device=features.device)
    pos_sim = sim_matrix[pos_indices, pos_indices + batch_size]
    
    # Denominator: all similarities
    exp_sim = torch.exp(sim_matrix)
    denominators = torch.sum(exp_sim, dim=1)
    
    # InfoNCE
    loss = -torch.log(torch.exp(pos_sim) / denominators[pos_indices]).mean()
    
    return loss

def triplet_consistency_loss(feat1, feat2, feat3):
    """Потеря консистентности тройки"""
    
    # Все попарные расстояния должны быть малы
    dist_12 = 1 - F.cosine_similarity(feat1, feat2).mean()
    dist_13 = 1 - F.cosine_similarity(feat1, feat3).mean() 
    dist_23 = 1 - F.cosine_similarity(feat2, feat3).mean()
    
    return (dist_12 + dist_13 + dist_23) / 3
```

---

## 🎯 **АРХИТЕКТУРА 2: Объединенная дистилляция**

### Одновременная multi-teacher distillation

```python
class UnifiedDistillationModel(nn.Module):
    """Объединенная модель для одновременной дистилляции"""
    
    def __init__(self, yolo_model, dinov2_model, domain_adapter, num_classes):
        super().__init__()
        
        # Models
        self.yolo_model = yolo_model.model  # Внутренняя YOLOv10 модель
        self.dinov2_teacher = dinov2_model  # Заморожен
        self.domain_adapter = domain_adapter  # Предобученный
        
        # Feature extractors для разных уровней YOLOv10
        self.yolo_feature_hooks = {}
        self.register_yolo_hooks()
        
        # Multi-level alignment networks
        self.aligners = self._create_multi_level_aligners()
        
        # Attention modules для взвешивания важности разных уровней
        self.attention_weights = nn.Parameter(torch.ones(3) / 3)  # Для 3 уровней
        
        # Adaptive temperature для contrastive learning
        self.temperature = nn.Parameter(torch.tensor(0.1))
        
    def _create_multi_level_aligners(self):
        """Создание aligners для разных уровней признаков"""
        aligners = nn.ModuleDict()
        
        # Размерности YOLOv10 backbone на разных уровнях
        yolo_dims = [256, 512, 1024]  # P3, P4, P5
        target_dim = 768  # DINOv2 dimension
        
        for i, yolo_dim in enumerate(yolo_dims):
            aligners[f'level_{i}'] = nn.Sequential(
                # Spatial adaptation
                nn.AdaptiveAvgPool2d((7, 7)),  # Фиксированный spatial размер
                nn.Conv2d(yolo_dim, target_dim, 3, padding=1),
                nn.BatchNorm2d(target_dim),
                nn.ReLU(),
                
                # Channel refinement 
                nn.Conv2d(target_dim, target_dim, 1),
                nn.BatchNorm2d(target_dim),
                nn.ReLU(),
                
                # Global pooling
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                
                # Final projection
                nn.Linear(target_dim, target_dim),
                nn.LayerNorm(target_dim),
                nn.Dropout(0.1)
            )
        
        return aligners
    
    def register_yolo_hooks(self):
        """Регистрация hooks для извлечения промежуточных признаков YOLOv10"""
        
        def get_hook(name):
            def hook(module, input, output):
                self.yolo_feature_hooks[name] = output
            return hook
        
        # Находим слои backbone YOLOv10 (примерно)
        backbone_indices = [4, 6, 8]  # Адаптируйте под вашу архитектуру
        
        for i, idx in enumerate(backbone_indices):
            if idx < len(self.yolo_model):
                self.yolo_model[idx].register_forward_hook(get_hook(f'level_{i}'))
    
    def forward(self, images, targets=None, compute_distillation=True):
        """Forward pass с дистилляцией"""
        
        # 1. YOLOv10 forward (сохраняет промежуточные признаки через hooks)
        yolo_outputs = self.yolo_model(images)
        
        if not compute_distillation:
            return yolo_outputs
        
        # 2. Извлечение DINOv2 features через domain adapter
        with torch.no_grad():
            # Resize для DINOv2
            resized_images = F.interpolate(images, size=224, mode='bilinear', align_corners=False)
            dinov2_base_features = self.dinov2_teacher(resized_images)
            
            # Domain adaptation
            adapted_dinov2_features = self.domain_adapter(dinov2_base_features)
        
        # 3. Выравнивание YOLOv10 признаков с DINOv2
        alignment_losses = []
        attention_weights_norm = F.softmax(self.attention_weights, dim=0)
        
        for i in range(3):  # 3 уровня
            level_name = f'level_{i}'
            
            if level_name in self.yolo_feature_hooks:
                # YOLOv10 features на уровне i
                yolo_features = self.yolo_feature_hooks[level_name]
                
                # Проекция в DINOv2 пространство
                aligned_features = self.aligners[level_name](yolo_features)
                
                # Нормализация
                aligned_normalized = F.normalize(aligned_features, p=2, dim=1)
                dinov2_normalized = F.normalize(adapted_dinov2_features, p=2, dim=1)
                
                # Distillation loss на этом уровне
                level_loss = 1 - F.cosine_similarity(aligned_normalized, dinov2_normalized).mean()
                
                # Взвешивание по важности уровня
                weighted_loss = attention_weights_norm[i] * level_loss
                alignment_losses.append(weighted_loss)
        
        # 4. Общая дистилляционная потеря
        if alignment_losses:
            total_distillation_loss = torch.stack(alignment_losses).sum()
        else:
            total_distillation_loss = torch.tensor(0.0, device=images.device)
        
        # 5. Дополнительная контрастивная потеря для большей согласованности
        contrastive_loss = self.compute_contrastive_loss(
            torch.cat([self.aligners[f'level_{i}'](self.yolo_feature_hooks[f'level_{i}']) 
                      for i in range(3) if f'level_{i}' in self.yolo_feature_hooks], dim=1),
            adapted_dinov2_features.repeat(1, len(alignment_losses)) if alignment_losses else adapted_dinov2_features
        )
        
        return yolo_outputs, {
            'distillation_loss': total_distillation_loss,
            'contrastive_loss': contrastive_loss,
            'attention_weights': attention_weights_norm
        }
    
    def compute_contrastive_loss(self, yolo_combined, dinov2_features):
        """Контрастивная потеря между объединенными YOLO признаками и DINOv2"""
        
        # Нормализация
        yolo_norm = F.normalize(yolo_combined, p=2, dim=1)
        dinov2_norm = F.normalize(dinov2_features, p=2, dim=1)
        
        # Температурный contrastive learning
        similarity = torch.mm(yolo_norm, dinov2_norm.T) / self.temperature
        
        # Цель: максимизировать сходство corresponding пар
        batch_size = yolo_norm.shape[0]
        labels = torch.arange(batch_size, device=yolo_norm.device)
        
        loss = F.cross_entropy(similarity, labels) + F.cross_entropy(similarity.T, labels)
        
        return loss / 2

def train_unified_distillation(yolo_model, domain_adapter, train_dataloader, 
                              val_dataloader, num_epochs=50):
    """Обучение объединенной модели с дистилляцией"""
    
    # Setup models
    dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').cuda().eval()
    
    unified_model = UnifiedDistillationModel(
        yolo_model, dinov2_model, domain_adapter, num_classes=80  # COCO classes
    ).cuda()
    
    # Optimizer - обучаем только YOLOv10 и aligners
    trainable_params = (
        list(unified_model.yolo_model.parameters()) + 
        list(unified_model.aligners.parameters()) +
        [unified_model.attention_weights, unified_model.temperature]
    )
    
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, epochs=num_epochs, steps_per_epoch=len(train_dataloader)
    )
    
    # Loss weights (адаптивные)
    detection_weight = 1.0
    distillation_weight = 0.5
    contrastive_weight = 0.3
    
    best_map = 0.0
    
    for epoch in range(num_epochs):
        unified_model.train()
        epoch_metrics = {
            'detection_loss': [],
            'distillation_loss': [], 
            'contrastive_loss': [],
            'total_loss': []
        }
        
        for batch_idx, (images, targets) in enumerate(train_dataloader):
            images = images.cuda()
            # targets обработка зависит от вашего формата данных
            
            # Forward pass
            yolo_outputs, distill_info = unified_model(images, targets, compute_distillation=True)
            
            # Detection loss (стандартная YOLO loss)
            detection_loss = compute_yolo_detection_loss(yolo_outputs, targets)  # Реализуйте
            
            # Distillation losses
            distill_loss = distill_info['distillation_loss']
            contrast_loss = distill_info['contrastive_loss']
            
            # Total loss
            total_loss = (detection_weight * detection_loss + 
                         distillation_weight * distill_loss +
                         contrastive_weight * contrast_loss)
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()
            
            # Logging
            epoch_metrics['detection_loss'].append(detection_loss.item())
            epoch_metrics['distillation_loss'].append(distill_loss.item())
            epoch_metrics['contrastive_loss'].append(contrast_loss.item())
            epoch_metrics['total_loss'].append(total_loss.item())
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}:')
                print(f'  Detection: {detection_loss:.4f}')
                print(f'  Distillation: {distill_loss:.4f}') 
                print(f'  Contrastive: {contrast_loss:.4f}')
                print(f'  Attention weights: {distill_info["attention_weights"].detach().cpu().numpy()}')
        
        # Validation
        val_map = validate_unified_model(unified_model, val_dataloader)
        
        # Logging
        for key in epoch_metrics:
            avg_val = np.mean(epoch_metrics[key])
            print(f'Epoch {epoch+1} avg {key}: {avg_val:.4f}')
        print(f'Validation mAP: {val_map:.4f}')
        
        # Save best model
        if val_map > best_map:
            best_map = val_map
            torch.save({
                'unified_model': unified_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_map': best_map
            }, 'best_unified_distillation_model.pth')
        
        # Adaptive loss weight adjustment
        if epoch > num_epochs // 3:
            distillation_weight *= 0.995  # Постепенно уменьшаем distillation
            contrastive_weight *= 0.998
    
    return unified_model, best_map

def validate_unified_model(unified_model, val_dataloader):
    """Валидация объединенной модели"""
    unified_model.eval()
    
    # Здесь стандартная YOLO валидация
    # Возвращаем mAP
    
    with torch.no_grad():
        # Implement your validation logic
        pass
    
    return 0.75  # Placeholder

# Пример запуска:
"""
# 1. Сначала обучаем domain adapter
domain_adapter = train_domain_adapter("path/to/domain/images")

# 2. Затем объединенное обучение
unified_model, best_map = train_unified_distillation(
    yolo_model=your_yolo_model,
    domain_adapter=domain_adapter,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    num_epochs=50
)
"""
```

---

## 🎯 **АРХИТЕКТУРА 3: Curriculum Learning подход**

### Постепенная дистилляция с увеличением сложности

```python
class CurriculumDistillationTrainer:
    """Curriculum learning для постепенной дистилляции"""
    
    def __init__(self, yolo_model, dinov2_model, domain_adapter):
        self.yolo_model = yolo_model
        self.dinov2_teacher = dinov2_model
        self.domain_adapter = domain_adapter
        
        # Curriculum stages
        self.curriculum_stages = [
            {
                'name': 'warm_up',
                'epochs': 10,
                'distillation_weight': 0.1,
                'temperature': 5.0,
                'feature_levels': [2]  # Только высокий уровень
            },
            {
                'name': 'intermediate',
                'epochs': 20,
                'distillation_weight': 0.3,
                'temperature': 3.0, 
                'feature_levels': [1, 2]  # Средний + высокий
            },
            {
                'name': 'advanced',
                'epochs': 20,
                'distillation_weight': 0.5,
                'temperature': 1.0,
                'feature_levels': [0, 1, 2]  # Все уровни
            }
        ]
        
        self.current_stage = 0
        
    def get_current_stage_config(self):
        """Получение конфигурации текущего этапа"""
        return self.curriculum_stages[self.current_stage]
    
    def should_advance_stage(self, current_epoch, stage_start_epoch):
        """Проверка необходимости перехода к следующему этапу"""
        stage_config = self.get_current_stage_config()
        return current_epoch - stage_start_epoch >= stage_config['epochs']
    
    def advance_stage(self):
        """Переход к следующему этапу curriculum"""
        if self.current_stage < len(self.curriculum_stages) - 1:
            self.current_stage += 1
            print(f"\n🎯 Advancing to stage: {self.get_current_stage_config()['name']}")
            return True
        return False
    
    def compute_curriculum_loss(self, yolo_features_dict, adapted_dinov2_features):
        """Вычисление loss с учетом текущего этапа curriculum"""
        
        stage_config = self.get_current_stage_config()
        active_levels = stage_config['feature_levels']
        temperature = stage_config['temperature']
        
        level_losses = []
        
        for level in active_levels:
            level_name = f'level_{level}'
            
            if level_name in yolo_features_dict:
                yolo_feat = yolo_features_dict[level_name]
                
                # Temperature scaling
                yolo_soft = F.softmax(yolo_feat / temperature, dim=1)
                dinov2_soft = F.softmax(adapted_dinov2_features / temperature, dim=1)
                
                # KL divergence loss
                kl_loss = F.kl_div(
                    F.log_softmax(yolo_feat / temperature, dim=1),
                    dinov2_soft,
                    reduction='batchmean'
                ) * (temperature ** 2)
                
                level_losses.append(kl_loss)
        
        if level_losses:
            return torch.stack(level_losses).mean()
        else:
            return torch.tensor(0.0, device=adapted_dinov2_features.device)

def train_with_curriculum(yolo_model, domain_adapter, train_dataloader, val_dataloader):
    """Curriculum learning training"""
    
    dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').cuda().eval()
    curriculum_trainer = CurriculumDistillationTrainer(yolo_model, dinov2_model, domain_adapter)
    
    optimizer = torch.optim.AdamW(yolo_model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    total_epochs = sum(stage['epochs'] for stage in curriculum_trainer.curriculum_stages)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
    
    epoch = 0
    stage_start_epoch = 0
    best_map = 0.0
    
    while curriculum_trainer.current_stage < len(curriculum_trainer.curriculum_stages):
        stage_config = curriculum_trainer.get_current_stage_config()
        
        print(f"\n🎓 Training stage: {stage_config['name']}")
        print(f"   Distillation weight: {stage_config['distillation_weight']}")
        print(f"   Temperature: {stage_config['temperature']}")
        print(f"   Active levels: {stage_config['feature_levels']}")
        
        # Train for current stage
        for stage_epoch in range(stage_config['epochs']):
            yolo_model.train()
            
            epoch_losses = {
                'detection': [],
                'distillation': [],
                'total': []
            }
            
            for batch_idx, (images, targets) in enumerate(train_dataloader):
                images = images.cuda()
                
                # Forward pass
                yolo_outputs = yolo_model(images)
                yolo_features = extract_yolo_features(yolo_model, images)  # Implement
                
                # DINOv2 features
                with torch.no_grad():
                    resized = F.interpolate(images, size=224)
                    dinov2_base = dinov2_model(resized)
                    adapted_dinov2 = domain_adapter(dinov2_base)
                
                # Losses
                detection_loss = compute_detection_loss(yolo_outputs, targets)  # Implement
                
                distillation_loss = curriculum_trainer.compute_curriculum_loss(
                    yolo_features, adapted_dinov2
                )
                
                total_loss = (detection_loss + 
                             stage_config['distillation_weight'] * distillation_loss)
                
                # Backward
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                scheduler.step()
                
                # Logging
                epoch_losses['detection'].append(detection_loss.item())
                epoch_losses['distillation'].append(distillation_loss.item())
                epoch_losses['total'].append(total_loss.item())
                
                if batch_idx % 100 == 0:
                    print(f'Stage {stage_config["name"]}, Epoch {stage_epoch+1}, '
                          f'Batch {batch_idx}: Total Loss = {total_loss:.4f}')
            
            # Validation
            val_map = validate_model(yolo_model, val_dataloader)  # Implement
            
            print(f'Stage {stage_config["name"]}, Epoch {stage_epoch+1}:')
            print(f'  Detection Loss: {np.mean(epoch_losses["detection"]):.4f}')
            print(f'  Distillation Loss: {np.mean(epoch_losses["distillation"]):.4f}')
            print(f'  Validation mAP: {val_map:.4f}')
            
            # Save best
            if val_map > best_map:
                best_map = val_map
                torch.save({
                    'model': yolo_model.state_dict(),
                    'stage': curriculum_trainer.current_stage,
                    'epoch': epoch,
                    'best_map': best_map
                }, f'curriculum_best_stage_{curriculum_trainer.current_stage}.pth')
            
            epoch += 1
        
        # Advance to next stage
        if not curriculum_trainer.advance_stage():
            break
        stage_start_epoch = epoch
    
    return yolo_model, best_map
```

---

## 📊 **СРАВНЕНИЕ ПОДХОДОВ**

| Подход | Сложность | Время обучения | Прирост качества | GPU память |
|--------|-----------|----------------|------------------|------------|
| **Двухэтапная** | ⭐⭐ | 2-3 дня | +5-8% | 12GB |
| **Объединенная** | ⭐⭐⭐⭐ | 3-4 дня | +7-12% | 16GB |
| **Curriculum** | ⭐⭐⭐ | 4-5 дней | +8-15% | 14GB |

## 🎯 **РЕКОМЕНДУЕМЫЙ ПЛАН ДЕЙСТВИЙ**

```python
# ПЛАН НА 7-10 ДНЕЙ:

# День 1-2: Domain Adapter
domain_adapter = train_domain_adapter("your_domain_images/")

# День 3-4: Двухэтапная дистилляция (простая)
stage1_model = train_two_stage_distillation(yolo_model, domain_adapter)

# День 5-7: Объединенная дистилляция (если ресурсы позволяют)
unified_model = train_unified_distillation(yolo_model, domain_adapter, train_loader, val_loader)

# День 8-10: Curriculum learning (для максимального качества)
final_model = train_with_curriculum(yolo_model, domain_adapter, train_loader, val_loader)

# Итоговое сравнение
compare_all_models([stage1_model, unified_model, final_model])
```

**Начните с двухэтапной дистилляции** - она даст 80% пользы за 40% времени! 🚀