import requests
import json
import base64
import os
from PIL import Image
import cv2
import numpy as np
from typing import List, Dict, Any
import time


class QwenVLAutoLabeler:
    def __init__(self, api_key: str, api_url: str = "http://5.35.4.197:8000/v1/chat/completions"):
        """
        Инициализация Qwen-VL API клиента
        
        Args:
            api_key: API ключ от Alibaba Cloud
            api_url: URL endpoint API
        """
        self.api_key = api_key
        self.api_url = api_url
        self.headers = {
            "Content-Type": "application/json",
        }
    
    def encode_image_to_base64(self, img) -> str:
        """
        Кодирование изображения в base64
        
        Args:
            image_path: путь к изображению
            
        Returns:
            base64 строка изображения
        """
        
        success, buffer = cv2.imencode('.jpg', img)

        encoded_image = base64.b64encode(buffer).decode("utf-8")
        
        return encoded_image
    
    def analyze_image(self, image, prompt: str) -> Dict[str, Any]:
        """
        Анализ изображения с помощью Qwen-VL API
        
        Args:
            image_path: путь к изображению
            prompt: промпт для анализа
            
        Returns:
            словарь с результатами анализа
        """
        # Кодируем изображение
        
        image_base64 = self.encode_image_to_base64(image)
        # Формируем запрос
        data = {
            "model": "Qwen3",  # или "qwen3" если точно поддерживается
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            "max_tokens": 2048,
            "temperature": 0.1
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=data,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                return result
            else:
                print(f"Ошибка API: {response.status_code}")
                
                return None
                
        except Exception as e:
            print(f"Ошибка при запросе: {e}")
            return None
    
    def detect_objects(self, image) -> List[Dict[str, Any]]:
        """
        Детекция объектов на изображении
        
        Args:
            image_path: путь к изображению
            
        Returns:
            список обнаруженных объектов
        """
        prompt = """
        Проанализируй это изображение и локализуй все контейнеры расположенные длинной стороной к камере. Даже те что видны частично
        
        Для каждого объекта укажи:
        1. Название объекта
        2. Приблизительные координаты bounding box в формате [x1, y1, x2, y2]
        3. уровень в стопке -1,2 или 3
        3. Уверенность в определении (высокая, средняя, низкая)
        
        Ответ предоставь в формате JSON:
        {
            "objects": [
                {
                    "name": "название объекта",
                    "bbox": [x1, y1, x2, y2],
                    "level": 1
                    "confidence": "уровень уверенности"
                }
            ]
        }
        """
        
        result = self.analyze_image(image, prompt)
        
        if result:
            try:
                # Парсим JSON из текстового ответа
                
                response_text = result["choices"][0]["message"]["content"].strip()
                # Ищем JSON в тексте ответа
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = response_text[start_idx:end_idx]
                    objects_data = json.loads(json_str)
                    return objects_data.get("objects", [])
            except Exception as e:
                print(f"Ошибка парсинга JSON: {e}")
                print(f"Ответ модели: {response_text}")
        
        return []
    
    def detect_number(self, image) -> List[Dict[str, Any]]:
        """
        Детекция объектов на изображении
        
        Args:
            image_path: путь к изображению
            
        Returns:
            список обнаруженных объектов
        """
        prompt = """
        Проанализируй это изображение контейнера и локализуй номер контейнера в фомате ISO 6346 
        исключая дополнительную информацию вроде 22G1 но обязательно включая контрольную цифру
        
        Для номера контейнера укажи:
        1. Ориентацию (горизонтальный/вертикальный)
        2. Приблизительные координаты bounding box в формате [x1, y1, x2, y2]
        3. значение номера
        3. Уверенность в определении (высокая, средняя, низкая)
        
        Ответ предоставь в формате JSON:
        {
            "objects": [
                {
                    "orientation": "название объекта",
                    "bbox": [x1, y1, x2, y2],
                    "number": TKRU 222222 2
                    "confidence": "уровень уверенности"
                }
            ]
        }
        """
        
        result = self.analyze_image(image, prompt)
        
        if result:
            try:
                # Парсим JSON из текстового ответа
                
                response_text = result["choices"][0]["message"]["content"].strip()
                # Ищем JSON в тексте ответа
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = response_text[start_idx:end_idx]
                    objects_data = json.loads(json_str)
                    return objects_data.get("objects", [])
            except Exception as e:
                print(f"Ошибка парсинга JSON: {e}")
                print(f"Ответ модели: {response_text}")
        
        return []
    
    def classify_scene(self, image) -> Dict[str, Any]:
        """
        Классификация сцены на изображении
        
        Args:
            image_path: путь к изображению
            
        Returns:
            словарь с классификацией сцены
        """
        prompt = """
        Классифицируй эту сцену. Определи:
        1. Основную категорию (например: городской пейзаж, природа, интерьер и т.д.)
        2. Основные цвета
        3. Освещение (дневное, ночное, искусственное)
        4. Основные объекты в сцене
        5. Общее настроение сцены
        
        Ответ предоставь в формате JSON.
        """
        
        result = self.analyze_image(image, prompt)
        if result:
            try:
                response_text = result["choices"][0]["message"]["content"].strip()
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = response_text[start_idx:end_idx]
                    return json.loads(json_str)
            except Exception as e:
                print(f"Ошибка парсинга JSON: {e}")
        
        return {}
    
    def generate_caption(self, image) -> str:
        """
        Генерация описания для изображения
        
        Args:
            image_path: путь к изображению
            
        Returns:
            текстовое описание изображения
        """
        prompt = "Опиши это изображение подробно, включая все важные детали."
        
        result = self.analyze_image(image, prompt)
        if result:
            return result["choices"][0]["message"]["content"].strip()
        
        return "Не удалось сгенерировать описание"
    
    def visualize_detections(self, image, objects: List[Dict[str, Any]], output_path: str = None):
        """
        Визуализация обнаруженных объектов на изображении
        
        Args:
            image_path: путь к исходному изображению
            objects: список обнаруженных объектов
            output_path: путь для сохранения результата
        """
        # Загружаем изображение
        
        if image is None:
            print("Не удалось загрузить изображение")
            return
        imgsize_y=image.shape[0]
        imgsize_x=image.shape[1]
        
        # Рисуем bounding boxes
        for obj in objects:
            name = obj.get("name", "unknown")
            bbox = obj.get("bbox", [])
            confidence = obj.get("confidence", "unknown")
            
            if len(bbox) == 4:
                bbox=[float('0.'+str(x if x!=1000 else 999).zfill(3)) for x in bbox]
                x1, y1, x2, y2 =  bbox
                x1=int(imgsize_x*x1)
                x2=int(imgsize_x*x2)
                y1=int(imgsize_y*y1)
                y2=int(imgsize_y*y2)
                # Рисуем прямоугольник
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Добавляем текст с названием и уверенностью
                label = f"{name} ({confidence})"
                cv2.putText(image, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Сохраняем или показываем результат
        if output_path:
            cv2.imwrite(output_path, image)
            print(f"Результат сохранен в: {output_path}")
        else:
            cv2.imshow("Detections", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def main():
    """
    Пример использования автоматической разметки изображений
    """
    # Замените на ваш API ключ
    API_KEY = "your_api_key_here"
    
    # Инициализация лейблера
    labeler = QwenVLAutoLabeler(API_KEY)
    
    # Путь к изображению для анализа
    image_path = "/home/roman/projects/dinov2/out1.jpg"
    
    if not os.path.exists(image_path):
        print(f"Изображение {image_path} не найдено")
        return
    
    print("🔍 Начинаем автоматическую разметку изображения...")
    
    # 1. Детекция объектов
    print("1. Детекция объектов...")
    
    image=cv2.imread(image_path)
    objects = labeler.detect_objects(image)
    print(f"Обнаружено объектов: {len(objects)}")
    for obj in objects:
        print(f"  - {obj.get('name')}: {obj.get('bbox')} ({obj.get('confidence')})")
    time.sleep(10)
    #Детекция номера
    imgsize_x=image.shape[1]
    imgsize_y=image.shape[0]
    for obj in objects:
        bbox=[float('0.'+str(x if x!=1000 else 999).zfill(3)) for x in obj['bbox']]
        x1, y1, x2, y2 =  bbox
        x1=int(imgsize_x*x1)
        x2=int(imgsize_x*x2)
        y1=int(imgsize_y*y1)
        y2=int(imgsize_y*y2)
        cv2.imwrite('test.jpg',image[y1:y2,x1:x2])
        number=labeler.detect_number(image[y1:y2,x1:x2])
        print(number)
    # 2. Классификация сцены
    print("\n2. Классификация сцены...")
    scene_info = labeler.classify_scene(image)
    print("Информация о сцене:")
    for key, value in scene_info.items():
        print(f"  - {key}: {value}")
    
    # 3. Генерация описания
    print("\n3. Генерация описания...")
    caption = labeler.generate_caption(image)
    print(f"Описание: {caption}")
    
    # 4. Визуализация результатов
    print("\n4. Визуализация результатов...")
    output_path = "detection_result.jpg"
    labeler.visualize_detections(image_path, objects, output_path)
    
    # 5. Сохранение результатов в JSON
    results = {
        "image_path": image_path,
        "objects": objects,
        "scene_classification": scene_info,
        "caption": caption,
        "timestamp": time.time()
    }
    
    json_output_path = "labeling_results.json"
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Результаты сохранены в {json_output_path}")


def batch_process_images(images_folder: str, api_key: str):
    """
    Пакетная обработка изображений в папке
    
    Args:
        images_folder: путь к папке с изображениями
        api_key: API ключ
    """
    labeler = QwenVLAutoLabeler(api_key)
    
    # Поддерживаемые форматы изображений
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    all_results = {}
    
    for filename in os.listdir(images_folder):
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext in supported_formats:
            image_path = os.path.join(images_folder, filename)
            print(f"Обработка: {filename}")
            
            try:
                # Детекция объектов
                objects = labeler.detect_objects(image_path)
                
                # Классификация сцены
                scene_info = labeler.classify_scene(image_path)
                
                # Генерация описания
                caption = labeler.generate_caption(image_path)
                
                # Сохраняем результаты
                all_results[filename] = {
                    "objects": objects,
                    "scene_classification": scene_info,
                    "caption": caption,
                    "processed_at": time.time()
                }
                
                # Пауза между запросами чтобы не превысить лимиты API
                time.sleep(1)
                
            except Exception as e:
                print(f"Ошибка при обработке {filename}: {e}")
    
    # Сохраняем все результаты
    output_file = "batch_labeling_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"Пакетная обработка завершена. Результаты сохранены в {output_file}")


if __name__ == "__main__":
    # Пример использования для одного изображения
    main()
    
    # Пример пакетной обработки
    # batch_process_images("path/to/images/folder", "your_api_key_here")