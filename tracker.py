# 🎯 Добавление фильтрации по минимальным размерам

```python
import numpy as np
from collections import deque
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass
from typing import Tuple, List, Optional
from enum import Enum

class TrackState(Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    APPROACHING = "approaching"
    DEPARTING = "departing"
    STATIONARY = "stationary"
    OCCLUDED = "occluded"
    OUT_OF_FRAME = "out_of_frame"


@dataclass
class TrackConfig:
    """Конфигурация трекера"""
    history_len: int = 40
    min_hits: int = 3
    max_misses: int = 20
    fps: float = 4.0
    
    # ⭐ НОВОЕ: Минимальные размеры для создания трека
    min_box_width: Optional[float] = None   # None = без ограничений
    min_box_height: Optional[float] = None
    min_box_area: Optional[float] = None    # Альтернатива: минимальная площадь
    
    # ⭐ НОВОЕ: Относительные размеры (% от размера кадра)
    min_box_width_ratio: Optional[float] = None   # Например, 0.05 = 5% ширины кадра
    min_box_height_ratio: Optional[float] = None
    
    # ⭐ НОВОЕ: Поведение при обновлении существующих треков
    allow_shrinking_below_threshold: bool = True  # Разрешить уменьшение ниже порога
    delete_tracks_below_threshold: bool = False   # Удалять треки при уменьшении
    
    # Пороги совместимости
    max_x2_deviation: float = 0.15
    max_cy_deviation: float = 0.10
    min_width_ratio: float = 0.4
    max_width_ratio: float = 2.5
    
    # Веса для score
    weight_x2: float = 0.6
    weight_cy: float = 0.3
    weight_width: float = 0.1
    
    # Сглаживание
    alpha_position: float = 0.4
    alpha_velocity: float = 0.6
    
    # Kalman
    use_kalman: bool = True
    kalman_process_variance: float = 2.0
    kalman_measurement_variance: float = 8.0
    
    # Детекция состояний
    motion_threshold_px: float = 10.0
    min_history_for_state: int = 5
    
    # Границы кадра
    edge_proximity_threshold: float = 0.95
    out_of_frame_margin: float = 1.2
    
    # Окклюзия
    occlusion_tolerance_multiplier: float = 0.15


class SizeValidator:
    """
    ⭐ Валидатор размеров боксов с поддержкой абсолютных и относительных порогов.
    """
    def __init__(self, config: TrackConfig, frame_width: int, frame_height: int):
        self.config = config
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Вычисляем абсолютные пороги из относительных
        self.abs_min_width = self._compute_absolute_width()
        self.abs_min_height = self._compute_absolute_height()
        self.abs_min_area = config.min_box_area
    
    def _compute_absolute_width(self) -> Optional[float]:
        """Вычисляет абсолютную минимальную ширину"""
        if self.config.min_box_width is not None:
            return self.config.min_box_width
        
        if self.config.min_box_width_ratio is not None:
            return self.config.min_box_width_ratio * self.frame_width
        
        return None
    
    def _compute_absolute_height(self) -> Optional[float]:
        """Вычисляет абсолютную минимальную высоту"""
        if self.config.min_box_height is not None:
            return self.config.min_box_height
        
        if self.config.min_box_height_ratio is not None:
            return self.config.min_box_height_ratio * self.frame_height
        
        return None
    
    def is_valid_for_new_track(self, box: Tuple[float, ...]) -> Tuple[bool, str]:
        """
        Проверяет, подходит ли бокс для создания нового трека.
        
        Returns:
            (is_valid, reason) - валидность и причина отклонения
        """
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        area = width * height
        
        # Проверка базовой корректности
        if width <= 0 or height <= 0:
            return False, "invalid_dimensions"
        
        # Проверка минимальной ширины
        if self.abs_min_width is not None and width < self.abs_min_width:
            return False, f"width_too_small ({width:.1f} < {self.abs_min_width:.1f})"
        
        # Проверка минимальной высоты
        if self.abs_min_height is not None and height < self.abs_min_height:
            return False, f"height_too_small ({height:.1f} < {self.abs_min_height:.1f})"
        
        # Проверка минимальной площади
        if self.abs_min_area is not None and area < self.abs_min_area:
            return False, f"area_too_small ({area:.1f} < {self.abs_min_area:.1f})"
        
        return True, "ok"
    
    def is_valid_for_update(self, box: Tuple[float, ...]) -> Tuple[bool, str]:
        """
        Проверяет, подходит ли бокс для обновления существующего трека.
        
        ⭐ Может иметь более мягкие критерии, если allow_shrinking_below_threshold=True
        """
        if self.config.allow_shrinking_below_threshold:
            # Разрешаем любые размеры для обновления
            x1, y1, x2, y2 = box
            if x2 > x1 and y2 > y1:
                return True, "ok"
            return False, "invalid_dimensions"
        else:
            # Те же критерии, что и для новых треков
            return self.is_valid_for_new_track(box)
    
    def should_delete_track(self, box: Tuple[float, ...]) -> bool:
        """
        ⭐ Проверяет, нужно ли удалить трек из-за маленького размера.
        """
        if not self.config.delete_tracks_below_threshold:
            return False
        
        is_valid, _ = self.is_valid_for_new_track(box)
        return not is_valid
    
    def get_thresholds_info(self) -> dict:
        """Возвращает информацию о порогах для логирования"""
        return {
            'min_width_px': self.abs_min_width,
            'min_height_px': self.abs_min_height,
            'min_area_px2': self.abs_min_area,
            'width_ratio': self.config.min_box_width_ratio,
            'height_ratio': self.config.min_box_height_ratio,
            'allow_shrinking': self.config.allow_shrinking_below_threshold,
            'delete_on_shrink': self.config.delete_tracks_below_threshold
        }


class AdaptiveKalmanFilter1D:
    """Kalman с адаптивной measurement_variance"""
    def __init__(self, process_variance=2.0, measurement_variance=8.0):
        self.base_process_var = process_variance
        self.base_measurement_var = measurement_variance
        self.estimate = None
        self.error_estimate = 1.0
        self.missed_updates = 0
        
    def predict(self):
        if self.estimate is None:
            return None
        self.error_estimate += self.base_process_var
        self.missed_updates += 1
        return self.estimate
    
    def update(self, measurement):
        if self.estimate is None:
            self.estimate = measurement
            self.error_estimate = self.base_measurement_var
            self.missed_updates = 0
            return self.estimate
        
        adaptive_meas_var = self.base_measurement_var * (1 + 0.5 * self.missed_updates)
        error_prediction = self.error_estimate + self.base_process_var
        kalman_gain = error_prediction / (error_prediction + adaptive_meas_var)
        self.estimate = self.estimate + kalman_gain * (measurement - self.estimate)
        self.error_estimate = (1 - kalman_gain) * error_prediction
        self.missed_updates = 0
        return self.estimate


class VelocityEstimator:
    """Оценка скорости только по последовательным кадрам"""
    def __init__(self, alpha=0.6, max_history=5):
        self.alpha = alpha
        self.max_history = max_history
        self.velocity = 0.0
        self.recent_deltas = deque(maxlen=max_history)
    
    def update(self, current_value, previous_value, dt):
        if dt == 1:
            delta = current_value - previous_value
            self.recent_deltas.append(delta)
            self.velocity = self.alpha * delta + (1 - self.alpha) * self.velocity
        elif dt > 1:
            if len(self.recent_deltas) > 0:
                avg_delta = np.median(self.recent_deltas)
                self.velocity = self.alpha * avg_delta + (1 - self.alpha) * self.velocity
    
    def get_velocity(self):
        return self.velocity
    
    def predict(self, frames_ahead):
        safe_frames = min(frames_ahead, 3)
        return self.velocity * safe_frames


class ContainerTrack:
    def __init__(self, track_id: int, frame_id: int, box: Tuple[float, ...], 
                 frame_width: int, frame_height: int, config: TrackConfig,
                 size_validator: SizeValidator):  # ⭐ Добавлен валидатор
        self.id = track_id
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.config = config
        self.size_validator = size_validator  # ⭐
        
        self.history = deque(maxlen=config.history_len)
        
        self.state = TrackState.TENTATIVE
        self.missed_frames = 0
        self.hit_streak = 0
        self.age = 0
        
        # ⭐ Флаг для отслеживания нарушения размеров
        self.size_violation_count = 0
        
        if config.use_kalman:
            self.kf_x2 = AdaptiveKalmanFilter1D(
                config.kalman_process_variance,
                config.kalman_measurement_variance
            )
            self.kf_cy = AdaptiveKalmanFilter1D(
                config.kalman_process_variance,
                config.kalman_measurement_variance
            )
        else:
            self.kf_x2 = None
            self.kf_cy = None
        
        self.smooth_width = None
        self.smooth_height = None
        
        self.velocity_estimator_x2 = VelocityEstimator(config.alpha_velocity)
        self.velocity_estimator_cy = VelocityEstimator(config.alpha_velocity)
        
        self.add_observation(frame_id, box)
    
    def add_observation(self, frame_id: int, box: Tuple[float, ...]):
        """⭐ Добавляет наблюдение с проверкой размеров"""
        x1, y1, x2, y2 = box
        
        if x2 <= x1 or y2 <= y1:
            return False
        
        # ⭐ Проверка размеров для обновления
        is_valid, reason = self.size_validator.is_valid_for_update(box)
        if not is_valid:
            self.size_violation_count += 1
            # Можно логировать или обрабатывать по-другому
            if self.config.delete_tracks_below_threshold:
                return False
        else:
            self.size_violation_count = 0  # Сброс счётчика
        
        width = x2 - x1
        height = y2 - y1
        cy = (y1 + y2) / 2.0
        
        # Обновление позиции
        if self.config.use_kalman and self.kf_x2 is not None:
            smooth_x2 = self.kf_x2.update(x2)
            smooth_cy = self.kf_cy.update(cy)
        else:
            if self.smooth_width is None:
                smooth_x2 = x2
                smooth_cy = cy
            else:
                last = self.history[-1]
                alpha = self.config.alpha_position
                smooth_x2 = alpha * x2 + (1 - alpha) * last['smooth_x2']
                smooth_cy = alpha * cy + (1 - alpha) * last['smooth_cy']
        
        # EMA для размеров
        if self.smooth_width is None:
            self.smooth_width = width
            self.smooth_height = height
        else:
            alpha = self.config.alpha_position
            self.smooth_width = alpha * width + (1 - alpha) * self.smooth_width
            self.smooth_height = alpha * height + (1 - alpha) * self.smooth_height
        
        # Обновление скорости
        if len(self.history) > 0:
            last = self.history[-1]
            dt = frame_id - last['frame']
            self.velocity_estimator_x2.update(x2, last['x2'], dt)
            self.velocity_estimator_cy.update(cy, last['cy'], dt)
        
        self.history.append({
            'frame': frame_id,
            'x2': x2,
            'cy': cy,
            'width': width,
            'height': height,
            'smooth_x2': smooth_x2,
            'smooth_cy': smooth_cy,
            'smooth_width': self.smooth_width,
            'smooth_height': self.smooth_height
        })
        
        self.hit_streak += 1
        self.missed_frames = 0
        self.age += 1
        
        return True
    
    def should_be_deleted_due_to_size(self) -> bool:
        """⭐ Проверяет, нужно ли удалить трек из-за размера"""
        if not self.config.delete_tracks_below_threshold:
            return False
        
        if not self.history:
            return False
        
        last = self.history[-1]
        box = (
            last['smooth_x2'] - last['smooth_width'],
            last['smooth_cy'] - last['smooth_height'] / 2,
            last['smooth_x2'],
            last['smooth_cy'] + last['smooth_height'] / 2
        )
        
        return self.size_validator.should_delete_track(box)
    
    def predict_position(self, frames_ahead: int = 1) -> Tuple[float, float, float, float]:
        """Прогноз позиции"""
        if not self.history:
            return (0, 0, 0, 0)
        
        last = self.history[-1]
        pred_x2 = last['smooth_x2'] + self.velocity_estimator_x2.predict(frames_ahead)
        pred_cy = last['smooth_cy'] + self.velocity_estimator_cy.predict(frames_ahead)
        pred_width = last['smooth_width']
        pred_height = last['smooth_height']
        
        return (pred_x2, pred_cy, pred_width, pred_height)
    
    def is_near_edge(self) -> bool:
        """Проверка близости к краю кадра"""
        if not self.history:
            return False
        
        last = self.history[-1]
        x2 = last['smooth_x2']
        threshold = self.config.edge_proximity_threshold * self.frame_width
        return x2 > threshold or x2 < (1 - self.config.edge_proximity_threshold) * self.frame_width
    
    def is_out_of_frame(self) -> bool:
        """Проверка выхода за пределы кадра"""
        pred_x2, pred_cy, pred_width, _ = self.predict_position(self.missed_frames)
        margin = self.config.out_of_frame_margin
        
        if pred_x2 < -pred_width * 0.5:
            return True
        if pred_x2 > self.frame_width * margin:
            return True
        if pred_cy < -pred_width * 0.5 or pred_cy > self.frame_height * margin:
            return True
        
        return False
    
    def get_smoothed_box(self, use_prediction: bool = False) -> Tuple[float, ...]:
        """Возвращает сглаженный бокс"""
        if not self.history:
            return (0, 0, 0, 0)
        
        if use_prediction and self.missed_frames > 0:
            x2, cy, width, height = self.predict_position(self.missed_frames)
        else:
            last = self.history[-1]
            x2 = last['smooth_x2']
            cy = last['smooth_cy']
            width = last['smooth_width']
            height = last['smooth_height']
        
        width = max(10, min(width, self.frame_width * 0.5))
        height = max(10, min(height, self.frame_height * 0.5))
        
        x1 = max(0, x2 - width)
        x2 = min(self.frame_width, x2)
        y1 = max(0, cy - height / 2)
        y2 = min(self.frame_height, cy + height / 2)
        
        return (x1, y1, x2, y2)
    
    def compatibility_score(self, new_box: Tuple[float, ...]) -> float:
        """Вычисляет совместимость"""
        if not self.history:
            return 0.0
        
        x1, y1, x2, y2 = new_box
        width = x2 - x1
        cy = (y1 + y2) / 2.0
        
        pred_x2, pred_cy, pred_width, _ = self.predict_position(self.missed_frames + 1)
        
        occlusion_factor = 1 + self.config.occlusion_tolerance_multiplier * self.missed_frames
        adaptive_x2_dev = self.config.max_x2_deviation * occlusion_factor
        adaptive_cy_dev = self.config.max_cy_deviation * occlusion_factor
        
        norm_x2_dev = abs(x2 - pred_x2) / self.frame_width
        norm_cy_dev = abs(cy - pred_cy) / self.frame_height
        width_ratio = width / (pred_width + 1e-6)
        
        if self.is_near_edge():
            penalty_x2 = 0.0
        else:
            penalty_x2 = min(1.0, norm_x2_dev / adaptive_x2_dev)
        
        penalty_cy = min(1.0, norm_cy_dev / adaptive_cy_dev)
        
        if self.config.min_width_ratio < width_ratio < self.config.max_width_ratio:
            penalty_width = 0.0
        else:
            if width_ratio <= self.config.min_width_ratio:
                penalty_width = 1.0 - (width_ratio / self.config.min_width_ratio)
            else:
                penalty_width = (width_ratio - self.config.max_width_ratio) / self.config.max_width_ratio
            penalty_width = min(1.0, penalty_width)
        
        score = 1.0 - (
            self.config.weight_x2 * penalty_x2 +
            self.config.weight_cy * penalty_cy +
            self.config.weight_width * penalty_width
        )
        
        return max(0.0, score)
    
    def update_state(self):
        """Обновление состояния трека"""
        if self.missed_frames > 5 and self.is_out_of_frame():
            self.state = TrackState.OUT_OF_FRAME
            return
        
        if self.missed_frames > 0:
            self.state = TrackState.OCCLUDED
            return
        
        if len(self.history) < self.config.min_history_for_state:
            self.state = TrackState.TENTATIVE
            return
        
        if self.hit_streak < self.config.min_hits:
            self.state = TrackState.TENTATIVE
            return
        
        recent = list(self.history)[-self.config.min_history_for_state:]
        x2_values = [h['smooth_x2'] for h in recent]
        x = np.arange(len(x2_values))
        trend = np.polyfit(x, x2_values, 1)[0]
        
        threshold = self.config.motion_threshold_px
        if trend > threshold:
            self.state = TrackState.APPROACHING
        elif trend < -threshold:
            self.state = TrackState.DEPARTING
        else:
            self.state = TrackState.STATIONARY


class ContainerTracker:
    def __init__(self, frame_width: int, frame_height: int, 
                 config: Optional[TrackConfig] = None):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.config = config or TrackConfig()
        
        # ⭐ Создаём валидатор размеров
        self.size_validator = SizeValidator(self.config, frame_width, frame_height)
        
        self.tracks: List[ContainerTrack] = []
        self.next_id = 1
        
        self.frame_count = 0
        self.total_detections = 0
        
        # ⭐ Статистика отклонённых боксов
        self.rejected_boxes_log = []
        self.tentative_log = []
    
    def update(self, frame_id: int, boxes: List[Tuple[float, ...]], 
               include_tentative: bool = False,
               include_occluded: bool = False) -> List[Tuple[int, Tuple, str]]:
        """Обновление трекера"""
        self.frame_count += 1
        self.total_detections += len(boxes)
        
        # ⭐ Валидация и фильтрация боксов по размеру
        valid_boxes, rejected_boxes = self._validate_and_filter_boxes(boxes, frame_id)
        
        if len(valid_boxes) == 0:
            return self._handle_no_detections(frame_id, include_tentative, include_occluded)
        
        matched_tracks, unmatched_boxes = self._associate(valid_boxes)
        
        # Обновление сопоставленных треков
        for track_idx, box_idx in matched_tracks:
            self.tracks[track_idx].add_observation(frame_id, valid_boxes[box_idx])
        
        # ⭐ Создание новых треков только для валидных боксов
        for box_idx in unmatched_boxes:
            new_track = ContainerTrack(
                self.next_id, frame_id, valid_boxes[box_idx],
                self.frame_width, self.frame_height, self.config,
                self.size_validator  # ⭐ Передаём валидатор
            )
            self.tracks.append(new_track)
            self.next_id += 1
        
        self._update_all_states()
        self._prune_tracks()
        
        return self._get_output(include_tentative, include_occluded)
    
    def _validate_and_filter_boxes(self, boxes: List[Tuple[float, ...]], 
                                   frame_id: int) -> Tuple[List[Tuple[float, ...]], List[Tuple]]:
        """
        ⭐ Валидирует боксы и фильтрует по размеру.
        
        Returns:
            (valid_boxes, rejected_boxes_with_reasons)
        """
        valid_boxes = []
        rejected_boxes = []
        
        for box in boxes:
            if len(box) != 4:
                rejected_boxes.append((box, "invalid_format"))
                continue
            
            x1, y1, x2, y2 = box
            
            # Базовая проверка
            if not (x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0):
                rejected_boxes.append((box, "invalid_coordinates"))
                continue
            
            # ⭐ Проверка размеров для новых треков
            is_valid, reason = self.size_validator.is_valid_for_new_track(box)
            
            if is_valid:
                valid_boxes.append(box)
            else:
                rejected_boxes.append((box, reason))
                # Логирование отклонённых боксов
                self.rejected_boxes_log.append({
                    'frame': frame_id,
                    'box': box,
                    'reason': reason,
                    'width': x2 - x1,
                    'height': y2 - y1
                })
        
        return valid_boxes, rejected_boxes
    
    def _associate(self, boxes: List[Tuple[float, ...]]) -> Tuple[List[Tuple[int, int]], List[int]]:
        """Сопоставление через Hungarian algorithm"""
        if len(self.tracks) == 0:
            return [], list(range(len(boxes)))
        
        cost_matrix = np.zeros((len(boxes), len(self.tracks)))
        
        for i, box in enumerate(boxes):
            for j, track in enumerate(self.tracks):
                score = track.compatibility_score(box)
                cost_matrix[i, j] = 1.0 - score
        
        box_indices, track_indices = linear_sum_assignment(cost_matrix)
        
        matched_pairs = []
        matched_box_set = set()
        
        for box_idx, track_idx in zip(box_indices, track_indices):
            cost = cost_matrix[box_idx, track_idx]
            if cost < 0.75:
                matched_pairs.append((track_idx, box_idx))
                matched_box_set.add(box_idx)
        
        unmatched_boxes = [i for i in range(len(boxes)) if i not in matched_box_set]
        
        return matched_pairs, unmatched_boxes
    
    def _handle_no_detections(self, frame_id: int, 
                             include_tentative: bool,
                             include_occluded: bool) -> List[Tuple[int, Tuple, str]]:
        """Обработка кадра без детекций"""
        for track in self.tracks:
            track.missed_frames += 1
            
            if self.config.use_kalman and track.kf_x2 is not None:
                track.kf_x2.predict()
                track.kf_cy.predict()
            
            track.update_state()
        
        self._prune_tracks()
        return self._get_output(include_tentative, include_occluded)
    
    def _update_all_states(self):
        """Обновление состояний всех треков"""
        for track in self.tracks:
            track.update_state()
    
    def _prune_tracks(self):
        """⭐ Очистка с учётом размеров"""
        active_tracks = []
        
        for track in self.tracks:
            # Удаление по стандартным причинам
            if track.missed_frames > self.config.max_misses:
                continue
            if track.state == TrackState.OUT_OF_FRAME:
                continue
            
            # ⭐ Удаление из-за маленького размера
            if track.should_be_deleted_due_to_size():
                continue
            
            active_tracks.append(track)
        
        self.tracks = active_tracks
    
    def _get_output(self, include_tentative: bool = False,
                   include_occluded: bool = False) -> List[Tuple[int, Tuple, str]]:
        """Формирование выходных данных"""
        result = []
        
        for track in self.tracks:
            if track.state == TrackState.TENTATIVE:
                if include_tentative:
                    self.tentative_log.append((self.frame_count, track.id))
                else:
                    continue
            
            if track.state == TrackState.OCCLUDED and not include_occluded:
                continue
            
            if track.state == TrackState.OUT_OF_FRAME:
                continue
            
            box = track.get_smoothed_box(use_prediction=(track.missed_frames > 0))
            result.append((track.id, box, track.state.value))
        
        return result
    
    def get_statistics(self) -> dict:
        """⭐ Расширенная статистика с информацией о размерах"""
        state_counts = {}
        for state in TrackState:
            state_counts[state.value] = sum(1 for t in self.tracks if t.state == state)
        
        # Статистика отклонённых боксов
        rejection_reasons = {}
        for entry in self.rejected_boxes_log:
            reason = entry['reason']
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
        
        return {
            'total_tracks': len(self.tracks),
            'state_distribution': state_counts,
            'avg_track_age': np.mean([t.age for t in self.tracks]) if self.tracks else 0,
            'frames_processed': self.frame_count,
            'avg_detections_per_frame': self.total_detections / max(1, self.frame_count),
            'tentative_tracks_created': len(self.tentative_log),
            'rejected_boxes_total': len(self.rejected_boxes_log),
            'rejection_reasons': rejection_reasons,
            'size_thresholds': self.size_validator.get_thresholds_info()
        }
    
    def get_rejected_boxes_summary(self) -> dict:
        """⭐ Детальная информация об отклонённых боксах"""
        if not self.rejected_boxes_log:
            return {'total': 0, 'by_frame': {}, 'size_distribution': {}}
        
        by_frame = {}
        widths = []
        heights = []
        
        for entry in self.rejected_boxes_log:
            frame = entry['frame']
            by_frame[frame] = by_frame.get(frame, 0) + 1
            widths.append(entry['width'])
            heights.append(entry['height'])
        
        return {
            'total': len(self.rejected_boxes_log),
            'by_frame': by_frame,
            'size_distribution': {
                'width': {
                    'min': min(widths),
                    'max': max(widths),
                    'mean': np.mean(widths),
                    'median': np.median(widths)
                },
                'height': {
                    'min': min(heights),
                    'max': max(heights),
                    'mean': np.mean(heights),
                    'median': np.median(heights)
                }
            }
        }


# ⭐ Примеры использования с разными конфигурациями
if __name__ == "__main__":
    
    # ============================================================
    # Пример 1: Абсолютные пороги (в пикселях)
    # ============================================================
    print("=" * 60)
    print("Пример 1: Абсолютные пороги")
    print("=" * 60)
    
    config_absolute = TrackConfig(
        fps=4.0,
        min_hits=2,
        max_misses=12,
        
        # ⭐ Минимальные размеры в пикселях
        min_box_width=100,   # Минимум 100 пикселей ширины
        min_box_height=120,  # Минимум 120 пикселей высоты
        
        allow_shrinking_below_threshold=True,   # Разрешить уменьшение
        delete_tracks_below_threshold=False     # Не удалять при уменьшении
    )
    
    tracker1 = ContainerTracker(1920, 1080, config_absolute)
    
    # Тестовые боксы
    boxes_f1 = [
        (100, 200, 300, 440),   # 200x240 - OK
        (500, 200, 580, 290),   # 80x90 - TOO SMALL (отклонён)
        (700, 200, 900, 440),   # 200x240 - OK
    ]
    
    results = tracker1.update(1, boxes_f1, include_tentative=True)
    print(f"\nFrame 1: {len(results)} треков создано")
    for track_id, box, state in results:
        print(f"  Track {track_id}: width={box[2]-box[0]:.0f}, height={box[3]-box[1]:.0f}, state={state}")
    
    print(f"\nОтклонено боксов: {len(tracker1.rejected_boxes_log)}")
    for entry in tracker1.rejected_boxes_log:
        print(f"  {entry['reason']}: width={entry['width']:.0f}, height={entry['height']:.0f}")
    
    
    # ============================================================
    # Пример 2: Относительные пороги (% от размера кадра)
    # ============================================================
    print("\n" + "=" * 60)
    print("Пример 2: Относительные пороги")
    print("=" * 60)
    
    config_relative = TrackConfig(
        fps=4.0,
        min_hits=2,
        max_misses=12,
        
        # ⭐ Минимальные размеры в % от кадра
        min_box_width_ratio=0.05,   # 5% ширины кадра = 96 пикс при 1920
        min_box_height_ratio=0.08,  # 8% высоты кадра = 86 пикс при 1080
        
        allow_shrinking_below_threshold=True,
        delete_tracks_below_threshold=False
    )
    
    tracker2 = ContainerTracker(1920, 1080, config_relative)
    
    results = tracker2.update(1, boxes_f1, include_tentative=True)
    print(f"\nFrame 1: {len(results)} треков создано")
    print(f"Пороги: width >= {tracker2.size_validator.abs_min_width:.0f}px, "
          f"height >= {tracker2.size_validator.abs_min_height:.0f}px")
    
    
    # ============================================================
    # Пример 3: Минимальная площадь
    # ============================================================
    print("\n" + "=" * 60)
    print("Пример 3: Минимальная площадь")
    print("=" * 60)
    
    config_area = TrackConfig(
        fps=4.0,
        min_hits=2,
        max_misses=12,
        
        # ⭐ Минимальная площадь бокса
        min_box_area=15000,  # 15000 пикселей² (например, 100x150)
        
        allow_shrinking_below_threshold=True,
        delete_tracks_below_threshold=False
    )
    
    tracker3 = ContainerTracker(1920, 1080, config_area)
    
    results = tracker3.update(1, boxes_f1, include_tentative=True)
    print(f"\nFrame 1: {len(results)} треков создано")
    print(f"Порог площади: >= {config_area.min_box_area} px²")
    
    
    # ============================================================
    # Пример 4: Удаление треков при уменьшении
    # ============================================================
    print("\n" + "=" * 60)
    print("Пример 4: Удаление при уменьшении размера")
    print("=" * 60)
    
    config_strict = TrackConfig(
        fps=4.0,
        min_hits=2,
        max_misses=12,
        
        min_box_width=100,
        min_box_height=120,
        
        allow_shrinking_below_threshold=False,  # ⭐ Запретить уменьшение
        delete_tracks_below_threshold=True      # ⭐ Удалять при уменьшении
    )
    
    tracker4 = ContainerTracker(1920, 1080, config_strict)
    
    # Создаём трек с большим боксом
    boxes_f1_big = [(100, 200, 300, 440)]  # 200x240
    results = tracker4.update(1, boxes_f1_big)
    print(f"\nFrame 1: {len(results)} треков")
    
    # Обновляем маленьким боксом
    boxes_f2_small = [(100, 200, 180, 310)]  # 80x110 - меньше порога
    results = tracker4.update(2, boxes_f2_small)
    print(f"Frame 2 (маленький бокс): {len(results)} треков (трек должен быть удалён)")
    
    
    # ============================================================
    # Пример 5: Комбинированные пороги
    # ============================================================
    print("\n" + "=" * 60)
    print("Пример 5: Комбинированные условия")
    print("=" * 60)
    
    config_combined = TrackConfig(
        fps=4.0,
        min_hits=2,
        max_misses=12,
        
        # ⭐ И ширина, И высота, И площадь
        min_box_width=80,
        min_box_height=100,
        min_box_area=10000,
        
        allow_shrinking_below_threshold=True,
        delete_tracks_below_threshold=False
    )
    
    tracker5 = ContainerTracker(1920, 1080, config_combined)
    
    test_boxes = [
        (100, 200, 300, 440),   # 200x240, area=48000 - OK
        (500, 200, 590, 310),   # 90x110, area=9900 - FAIL (area)
        (700, 200, 850, 290),   # 150x90, area=13500 - FAIL (height)
    ]
    
    results = tracker5.update(1, test_boxes, include_tentative=True)
    print(f"\nFrame 1: {len(results)} треков создано из {len(test_boxes)} боксов")
    
    print("\nСтатистика:")
    stats = tracker5.get_statistics()
    print(f"  Отклонено боксов: {stats['rejected_boxes_total']}")
    print(f"  Причины отклонения: {stats['rejection_reasons']}")
    
    print("\nДетальная статистика отклонённых боксов:")
    rejected_summary = tracker5.get_rejected_boxes_summary()
    print(f"  Всего: {rejected_summary['total']}")
    print(f"  Размеры (ширина): min={rejected_summary['size_distribution']['width']['min']:.0f}, "
          f"mean={rejected_summary['size_distribution']['width']['mean']:.0f}")
    
    
    # ============================================================
    # Пример 6: Реальный сценарий - контейнеры 20ft/40ft
    # ============================================================
    print("\n" + "=" * 60)
    print("Пример 6: Реальный сценарий (контейнеры)")
    print("=" * 60)
    
    # Предположим:
    # - Камера 1920x1080
    # - Контейнер 40ft на расстоянии 10м занимает ~200x240 пикселей
    # - Контейнер 20ft на том же расстоянии ~100x240 пикселей
    # - Хотим отфильтровать мелкий мусор (коробки, люди и т.д.)
    
    config_real = TrackConfig(
        fps=4.0,
        min_hits=3,          # 3 подряд кадра для подтверждения
        max_misses=16,       # 4 секунды при 4 FPS
        
        # ⭐ Минимальные размеры для контейнера
        min_box_width=80,    # Минимум 80 пикселей (даже 20ft вдалеке)
        min_box_height=100,  # Минимум 100 пикселей высоты
        min_box_area=10000,  # Минимум 10000 px² (100x100)
        
        # Разрешаем временное уменьшение (окклюзия, перспектива)
        allow_shrinking_below_threshold=True,
        
        # Но удаляем если долго маленький (уехал вдаль)
        delete_tracks_below_threshold=False,
        
        # Настройки трекинга
        use_kalman=True,
        motion_threshold_px=8.0,
        max_x2_deviation=0.15,
        max_cy_deviation=0.10
    )
    
    tracker_real = ContainerTracker(1920, 1080, config_real)
    
    # Симуляция реального потока
    print("\nСимуляция 10 кадров:")
    
    frames_data = [
        # Frame 1: 2 контейнера + мусор
        [(100, 300, 280, 540),   # Контейнер 1: 180x240
         (500, 350, 650, 550),   # Контейнер 2: 150x200
         (800, 400, 850, 450)],  # Мусор: 50x50 - отклонён
        
        # Frame 2: движение
        [(120, 300, 300, 540),
         (520, 350, 670, 550)],
        
        # Frame 3: один контейнер окклюдирован
        [(140, 300, 320, 540)],
        
        # Frame 4: оба видны
        [(160, 300, 340, 540),
         (560, 350, 710, 550)],
        
        # Frame 5-10: продолжение движения
        [(180, 300, 360, 540), (580, 350, 730, 550)],
        [(200, 300, 380, 540), (600, 350, 750, 550)],
        [(220, 300, 400, 540), (620, 350, 770, 550)],
        [(240, 300, 420, 540), (640, 350, 790, 550)],
        [(260, 300, 440, 540), (660, 350, 810, 550)],
        [(280, 300, 460, 540), (680, 350, 830, 550)],
    ]
    
    for frame_id, boxes in enumerate(frames_data, start=1):
        results = tracker_real.update(frame_id, boxes, include_occluded=True)
        print(f"\n  Frame {frame_id}: {len(boxes)} детекций → {len(results)} треков")
        for track_id, box, state in results:
            w, h = box[2] - box[0], box[3] - box[1]
            print(f"    Track {track_id}: {w:.0f}x{h:.0f}, state={state}")
    
    print("\n" + "=" * 60)
    print("Финальная статистика:")
    print("=" * 60)
    final_stats = tracker_real.get_statistics()
    print(f"Всего кадров: {final_stats['frames_processed']}")
    print(f"Всего детекций: {tracker_real.total_detections}")
    print(f"Активных треков: {final_stats['total_tracks']}")
    print(f"Отклонено боксов: {final_stats['rejected_boxes_total']}")
    print(f"Причины отклонения: {final_stats['rejection_reasons']}")
    print(f"\nРаспределение по состояниям:")
    for state, count in final_stats['state_distribution'].items():
        if count > 0:
            print(f"  {state}: {count}")
    
    print(f"\nПороги размеров:")
    thresholds = final_stats['size_thresholds']
    print(f"  Мин. ширина: {thresholds['min_width_px']} px")
    print(f"  Мин. высота: {thresholds['min_height_px']} px")
    print(f"  Мин. площадь: {thresholds['min_area_px2']} px²")
```

---

## 🎯 Ключевые возможности

### 1. **Три типа порогов размеров**
```python
# Абсолютные (пиксели)
min_box_width=100
min_box_height=120

# Относительные (% от кадра)
min_box_width_ratio=0.05  # 5% ширины

# По площади
min_box_area=15000  # px²
```

### 2. **Гибкое поведение**
```python
# Разрешить временное уменьшение (окклюзия)
allow_shrinking_below_threshold=True

# Удалять треки при стойком уменьшении
delete_tracks_below_threshold=True
```

### 3. **Детальная статистика**
```python
stats = tracker.get_statistics()
# Включает:
# - Количество отклонённых боксов
# - Причины отклонения
# - Распределение размеров

rejected = tracker.get_rejected_boxes_summary()
# Детальный анализ отклонённых боксов
```

### 4. **Логирование для отладки**
```python
# Все отклонённые боксы сохраняются
tracker.rejected_boxes_log
# Содержит: frame, box, reason, width, height
```

---

## 🚀 Рекомендации по настройке

```python
# Для фильтрации мелкого мусора
config = TrackConfig(
    min_box_width=80,
    min_box_height=100,
    allow_shrinking_below_threshold=True,
    delete_tracks_below_threshold=False
)

# Для строгой фильтрации
config = TrackConfig(
    min_box_area=20000,  # Только крупные объекты
    allow_shrinking_below_threshold=False,
    delete_tracks_below_threshold=True
)

# Для адаптации к разным разрешениям
config = TrackConfig(
    min_box_width_ratio=0.04,  # 4% ширины кадра
    min_box_height_ratio=0.06  # 6% высоты кадра
)
```
