def detect_action(self, min_history_for_action: int = 6) -> Optional[str]:
    """
    Определяет, был ли контейнер:
      - "picked_up" (захвачен)
      - "placed" (поставлен)
      - None (неизвестно / недостаточно данных)
    
    Условия:
      - Захват: x2 > 2000 И width > 1400 И тренд x2 положительный
      - Постановка: тренд x2 отрицательный И стабилизация после снижения
    """
    if len(self.history) < min_history_for_action:
        return None

    # Извлекаем последние N точек
    recent = list(self.history)[-min_history_for_action:]
    
    x2_vals = np.array([h['x2'] for h in recent])
    y1_vals = np.array([h['smooth_cy'] - h['smooth_height'] / 2 for h in recent])
    width_vals = np.array([h['smooth_width'] for h in recent])

    # Тренды (линейная регрессия)
    frames = np.arange(len(x2_vals))
    trend_x2 = np.polyfit(frames, x2_vals, 1)[0]  # наклон
    trend_y1 = np.polyfit(frames, y1_vals, 1)[0]

    current_x2 = x2_vals[-1]
    current_width = width_vals[-1]

    # === Условие захвата ===
    if (
        current_x2 > 2000 and
        current_width > 1400 and
        trend_x2 > 5.0 and           # x2 растёт достаточно быстро (пикс/кадр)
        trend_y1 < -3.0              # y1 падает (подъём)
    ):
        return "picked_up"

    # === Условие постановки ===
    # Требуем: x2 уменьшается, y1 растёт, и последние 2-3 кадра стабильны
    if (
        trend_x2 < -5.0 and          # x2 уменьшается
        trend_y1 > 3.0               # y1 растёт (опускание)
    ):
        # Проверка стабилизации: последние 2 кадра почти не меняются
        if len(x2_vals) >= 3:
            last_diff = abs(x2_vals[-1] - x2_vals[-2])
            prev_diff = abs(x2_vals[-2] - x2_vals[-3])
            if last_diff < 8.0 and prev_diff < 8.0:
                return "placed"

    return None
