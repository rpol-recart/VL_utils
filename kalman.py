import numpy as np

class CryoliteRatioFilter:
    """
    Промышленный фильтр для оценки .
    
    Работает в реальном времени. Использует:
    - Частые, но шумные косвенные измерения (раз в 3 ч)
    - Редкие, но точные лабораторные данные (раз в 72 ч)
    
    Решает проблему "пилообразных скачков" за счёт оценки систематического смещения (Bias).
    Защищён от выбросов в лабораторных данных.
    """
    
    def __init__(self, initial_param, initial_bias=0.0):
        # Состояние: [Param, Bias]
        self.x = np.array([initial_param, initial_bias], dtype=float)
        
        # Начальная ковариация: высокая неуверенность в Bias
        self.P = np.diag([0.02, 0.05])
        
        # Модель процесса: Identity (величины меняются медленно)
        self.F = np.eye(2)
        
        # Процессный шум — КЛЮЧЕВОЙ ПАРАМЕТР
        # Q[0] — насколько быстро может меняться Param (рекомендуемый диапазон: 1e-4 ... 1e-3)
        # Q[1] — насколько быстро может дрейфовать Bias (очень медленно!)
        self.Q = np.diag([5e-4, 1e-7])
        
        # Шум измерений
        self.R_indirect = 0.03**2   # Дисперсия ~3% для косвенных
        self.R_lab = 0.005**2       # Дисперсия ~0.5% для лаборатории

    def update(self, measurement, is_lab=False, outlier_sigma=3.0):
        """
        Обновление фильтра новым измерением.
        
        Параметры:
        - measurement: float, измеренное значение Param
        - is_lab: bool, True если это лабораторный анализ
        - outlier_sigma: float, порог для отбраковки выбросов (в сигмах)
        
        Возвращает:
        - current_Param: текущая оценка Param (для оператора)
        """
        # === Шаг 1: Прогноз ===
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        # === Шаг 2: Подготовка к коррекции ===
        H = np.array([[1.0, 0.0]]) if is_lab else np.array([[1.0, 1.0]])
        R = self.R_lab if is_lab else self.R_indirect

        # === Шаг 3: Вычисление невязки ===
        residual = measurement - H @ self.x
        residual_cov = H @ self.P @ H.T + R

        # === Шаг 4: Защита от выбросов (только для лаборатории) ===
        should_update = True
        if is_lab:
            if np.abs(residual) > outlier_sigma * np.sqrt(residual_cov):
                should_update = False  # Пропускаем обновление

        # === Шаг 5: Коррекция (если не выброс) ===
        if should_update:
            kalman_gain = self.P @ H.T @ np.linalg.inv(residual_cov)
            self.x = self.x + kalman_gain * residual
            self.P = (np.eye(2) - np.outer(kalman_gain, H)) @ self.P

        return float(self.x[0])  # Возвращаем только Param для оператора

    def get_bias(self):
        """Возвращает текущую оценку систематической ошибки модели."""
        return float(self.x[1])
    
    def reset_bias(self, new_bias=0.0):
        """Сброс Bias (например, после калибровки датчиков)."""
        self.x[1] = new_bias
        self.P[1, 1] = 0.05  # Снова высокая неуверенность
