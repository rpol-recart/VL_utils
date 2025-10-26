from huggingface_hub import snapshot_download

#Qwen3-VL-8B-Instruct
# Пример: скачивание всей модели
snapshot_download(
    repo_id="Qwen/Qwen3-VL-2B-Instruct-FP8",  # например, "facebook/detr-resnet-50"
    local_dir="./Qwen3-2B_FP8",         # папка для сохранения
    local_dir_use_symlinks=False    # чтобы избежать симлинков (опционально)
)
