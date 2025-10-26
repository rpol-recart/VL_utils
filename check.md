Конечно! Ниже — **полностью обновлённая функция `osd_sink_pad_buffer_probe`**, интегрированная в ваш существующий DeepStream-пайплайн, с:

- валидацией по **ISO 6346**,
- проверкой формата: **первые 4 символа — буквы, 4-я — всегда `U`**,
- **последний символ — контрольная цифра (0–9)**,
- **коррекцией типичных ошибок** (`0↔O`, `1↔I`, `8↔B`, и т.д.),
- **отображением результата на OSD** и логированием.

---

### ✅ Обновлённая функция `osd_sink_pad_buffer_probe`

Замените **весь блок `osd_sink_pad_buffer_probe`** в вашем `deepstream_pipeline.py` на следующий код:

```python
def osd_sink_pad_buffer_probe(self, pad, info, u_data):
    """Анализ результатов детекции и OCR с постпроцессингом по ISO 6346"""
    
    def validate_iso6346(number: str):
        """Проверяет корректность контейнерного номера по ISO 6346."""
        if not number or len(number) < 11:
            return False, None
        clean = number.replace(' ', '').replace('-', '').upper()
        if len(clean) != 11:
            return False, None
        prefix = clean[:4]
        serial = clean[4:10]
        try:
            check_digit = int(clean[10])
        except ValueError:
            return False, None

        # Проверка формата: первые 4 — буквы, 4-я — 'U'
        if not prefix[:3].isalpha() or prefix[3] != 'U':
            return False, None

        # Проверка: серийный номер — только цифры
        if not serial.isdigit():
            return False, None

        char_values = {
            '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
            'A': 10, 'B': 12, 'C': 13, 'D': 14, 'E': 15, 'F': 16, 'G': 17, 'H': 18, 'I': 19,
            'J': 20, 'K': 21, 'L': 23, 'M': 24, 'N': 25, 'O': 26, 'P': 27, 'Q': 28, 'R': 29,
            'S': 30, 'T': 31, 'U': 32, 'V': 34, 'W': 35, 'X': 36, 'Y': 37, 'Z': 38
        }

        total = 0
        weights = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        for i, c in enumerate(prefix + serial):
            val = char_values.get(c, 0)
            total += val * weights[i]

        computed = total % 11
        if computed == 10:
            computed = 0

        return computed == check_digit, computed

    def generate_candidates(text):
        """Генерирует кандидаты с заменой похожих символов."""
        replacements = {
            '0': ['O'], 'O': ['0'],
            '1': ['I'], 'I': ['1'],
            '8': ['B'], 'B': ['8'],
            '6': ['G'], 'G': ['6'],
            '5': ['S'], 'S': ['5'],
            '2': ['Z'], 'Z': ['2'],
            '4': ['A'], 'A': ['4'],
        }
        candidates = {text}
        for i, c in enumerate(text):
            if c in replacements:
                new_candidates = set()
                for cand in candidates:
                    for alt in replacements[c]:
                        new_cand = cand[:i] + alt + cand[i+1:]
                        new_candidates.add(new_cand)
                candidates.update(new_candidates)
        return list(candidates)

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            detection_info = {
                'class_id': obj_meta.class_id,
                'confidence': obj_meta.confidence,
                'bbox': {
                    'left': obj_meta.rect_params.left,
                    'top': obj_meta.rect_params.top,
                    'width': obj_meta.rect_params.width,
                    'height': obj_meta.rect_params.height
                }
            }

            final_number = None
            display_text = "INVALID"

            if obj_meta.classifier_meta_list:
                l_classifier = obj_meta.classifier_meta_list
                while l_classifier is not None:
                    try:
                        classifier_meta = pyds.NvDsClassifierMeta.cast(l_classifier.data)
                        if classifier_meta.label_info_list:
                            l_label = classifier_meta.label_info_list
                            while l_label is not None:
                                try:
                                    label_info = pyds.NvDsLabelInfo.cast(l_label.data)
                                    raw_number = label_info.result_label.strip().upper()
                                    if not raw_number:
                                        continue

                                    # Валидация
                                    is_valid, _ = validate_iso6346(raw_number)
                                    if is_valid:
                                        final_number = raw_number.replace(' ', '').replace('-', '')
                                        display_text = f"{final_number} ✅"
                                    else:
                                        # Попытка коррекции
                                        candidates = generate_candidates(raw_number)
                                        for cand in candidates:
                                            if validate_iso6346(cand)[0]:
                                                clean_cand = cand.replace(' ', '').replace('-', '')
                                                final_number = clean_cand
                                                display_text = f"{clean_cand} ✅ (corrected)"
                                                break
                                        if not final_number:
                                            display_text = f"{raw_number} ❌"
                                except StopIteration:
                                    break
                                try:
                                    l_label = l_label.next
                                except StopIteration:
                                    break
                    except StopIteration:
                        break
                    try:
                        l_classifier = l_classifier.next
                    except StopIteration:
                        break

            # Обновляем отображаемый текст на OSD
            obj_meta.text_params.display_text = display_text

            # Логирование
            print(f"📦 Container detected: {display_text}")
            print(f"   Confidence: {detection_info['confidence']:.3f}")
            print(f"   BBox: {detection_info['bbox']}")

            # (Опционально) отправка в бэкенд
            if final_number:
                self.send_to_backend(frame_meta.source_id, final_number, detection_info['bbox'])

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        # FPS tracking
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            elapsed = time.time() - self.start_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0
            print(f"📊 Processing: {fps:.1f} FPS")

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK
```

---

### 🔧 Дополнительно: метод `send_to_backend` (если ещё не добавлен)

Добавьте в класс `ContainerDetectionPipeline`:

```python
def send_to_backend(self, camera_id, container_id, bbox):
    import json, time
    payload = {
        "timestamp": time.time(),
        "camera_id": camera_id,
        "container_id": container_id,
        "bbox": bbox
    }
    with open("/tmp/container_log.jsonl", "a") as f:
        f.write(json.dumps(payload) + "\n")
```

---

### ✅ Что делает этот код:

| Функция | Описание |
|--------|--------|
| **Формат-валидация** | Первые 4 символа — буквы, 4-й — `U`, серийный номер — 6 цифр, последний — цифра |
| **ISO 6346** | Проверка контрольной суммы |
| **Коррекция** | Автоматическая замена `0/O`, `1/I`, `8/B` и др. |
| **OSD** | Отображает ✅ / ❌ и скорректированный номер |
| **Логирование** | Печатает в консоль и сохраняет в файл |

---

Теперь ваша система будет **отбрасывать недопустимые номера** и **исправлять типичные OCR-ошибки**, строго соблюдая стандарт контейнерных номеров.

Если нужно — могу также сгенерировать **unit-тесты** для `validate_iso6346` или **интеграцию с MQTT**.
