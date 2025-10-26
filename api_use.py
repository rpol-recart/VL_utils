import os
import json
import requests
from pathlib import Path
import re

def format_string(input_str: str) -> str:
    return re.sub(r'^```json\s*|\s*```$', '', input_str, flags=re.DOTALL)

# === Настройки ===
image_dir = Path("./auto")  # ← замените на путь к вашей папке
output_file = "results2.json"
api_url = "http://localhost:8000/v1/chat/completions"
model_name = "My_Model"  # ← укажите имя модели, как она зарегистрирована в вашем API-сервере

# Поддерживаемые расширения
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

# Собираем все изображения
image_paths = [p for p in image_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTS]
print(f"Найдено изображений: {len(image_paths)}")

results = {}
encoded_image=[]
mime_type=[]
for img_path in sorted(image_paths):
    print(f"\nОбработка: {img_path.name}")

    # Чтение изображения и кодирование в base64
    import base64
    with open(img_path, "rb") as image_file:
        encoded_image.append(base64.b64encode(image_file.read()).decode("utf-8"))
    mime_type.append("image/jpeg" if img_path.suffix.lower() in {".jpg", ".jpeg"} else "image/png")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type[0]};base64,{encoded_image[0]}"
                }
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type[1]};base64,{encoded_image[1]}"
                }
            },
            {
                "type": "text",
                "text": (
                    'Сравни 2 изображения , что изменилось '
                    
                ),
            },
        ],
    }
]
payload = {
    "model": model_name,
    "messages": messages,
    "max_tokens": 256,
    "temperature": 0.3,
    "stream": False
}
headers = {
    "Content-Type": "application/json"
}
try:
    response = requests.post(api_url, json=payload, headers=headers, timeout=60)
    response.raise_for_status()
    output_text = response.json()["choices"][0]["message"]["content"].strip()
    print(output_text)
    # Очистка от markdown-обёртки (если есть)
    #clean_output = format_string(output_text)
    # Попытка распарсить JSON
    try:
        parsed = json.loads(output_text)
        print(parsed)
        results[str(img_path)] = parsed
        container_number = [x.get("number") for x in parsed] if isinstance(parsed, list) else None
    except (json.JSONDecodeError, TypeError):
        container_number = None
        print(f"⚠️  Не удалось распарсить JSON: {repr(clean_output)}")
        results[str(img_path)] = {"raw_output": clean_output, "error": "JSON decode failed"}
    print(f"✅ Результат: {container_number}")
except Exception as e:
        print(f"❌ Ошибка при обработке {img_path.name}: {e}")
        results[str(img_path)] = {"error": str(e)}

# === Сохранение результатов ===
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results4, f, indent=2, ensure_ascii=False)

print(f"\n✅ Все изображения обработаны. Результаты сохранены в {output_file}")
