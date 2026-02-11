# Qwen Bbox OCR

Веб-приложение для распознавания PDF через **vLLM** с моделью **Qwen3-VL** (или Qwen2.5-VL): загрузка PDF → конвертация в изображения → OCR по страницам → JSON-структура документа и Markdown. В интерфейсе отображаются страницы с **bbox-разметкой** (текст, таблицы, изображения, печати, подписи).

## Требования

- **Сервер**: Ubuntu 24.04, CUDA, Tesla A10 24 ГБ (или другой GPU с достаточным объёмом VRAM для Qwen-VL).
- **vLLM** с моделью Qwen-VL запускается **отдельно** (не в этом репозитории). Это приложение — только клиент: принимает PDF, конвертирует в картинки и дергает API vLLM.

## Архитектура

1. **Сервер vLLM** — поднимается отдельно на том же хосте или в сети. На нём развёрнута модель, например `Qwen/Qwen3-VL-235B-A22B` (или `Qwen/Qwen2.5-VL-7B-Instruct` для тестов).
2. **Сервис qwen-bbox-ocr** (этот проект) — FastAPI: загрузка PDF, конвертация в PNG по страницам, запросы к vLLM, возврат JSON + Markdown + base64 изображений страниц для отображения bbox в браузере.

Подключение к vLLM задаётся через **переменные окружения** (файл `.env`).

## Запуск vLLM отдельно (пример)

На сервере с GPU:

```bash
# Пример для Qwen2.5-VL (меньше модель для проверки)
pip install vllm
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --served-model-name Qwen/Qwen2.5-VL-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000
```

Для **Qwen3-VL-235B-A22B** используйте соответствующее имя модели и при необходимости tensor parallelism (несколько GPU).

## Проверка работы сервера vLLM (Qwen)

Ответ `{"detail":"Not Found"}` на запросы к `http://localhost:8000` или `http://localhost:8000/v1` по GET — нормален: это не страницы, а API. Проверять нужно так:

1. **Health-check** (сервер жив и готов принимать запросы):
   ```bash
   curl http://localhost:8000/health
   ```
   Ожидается ответ без ошибки (например `{"status":"ok"}` или пустое тело с кодом 200).

2. **Список загруженных моделей** (модель действительно поднята):
   ```bash
   curl http://localhost:8000/v1/models
   ```
   В ответе должен быть объект с полем `data` и списком моделей; имя модели должно совпадать с `VLLM_MODEL` в `.env` приложения qwen-bbox-ocr.

3. **Проверка vision (опционально)** — запрос chat completions с картинкой (тест того, что Qwen-VL принимает изображения):
   ```bash
   curl -X POST http://localhost:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model":"<имя_модели_из_v1/models>","messages":[{"role":"user","content":[{"type":"text","text":"What is in this image?"},{"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="}}]}],"max_tokens":50}'
   ```
   Подставьте реальное имя модели из вывода `curl http://localhost:8000/v1/models`.

Если `curl http://localhost:8000/health` не отвечает или соединение отклонено — vLLM не запущен или слушает другой порт/хост. Если `/v1/models` возвращает пустой список — модель ещё не загрузилась (подождите или проверьте логи запуска vLLM).

## Запуск веб-приложения (Docker)

1. Скопируйте переменные окружения и задайте URL vLLM:

   ```bash
   cp .env.example .env
   # Отредактируйте .env: VLLM_BASE_URL и при необходимости VLLM_MODEL
   ```

2. Запуск через docker-compose:

   ```bash
   docker-compose up -d --build
   ```

3. Интерфейс: **http://localhost:8010** (или хост:8010). Загрузите PDF и нажмите «Распознать».

Если vLLM крутится на том же хосте, в `.env` укажите:

- **Linux**: `VLLM_BASE_URL=http://172.17.0.1:8000/v1` (IP docker0) или при запуске vLLM в другой сети — соответствующий IP/хост.
- **Windows/Mac**: `VLLM_BASE_URL=http://host.docker.internal:8000/v1` (уже подставлено в `docker-compose.yml` по умолчанию).

## Переменные окружения (.env)

| Переменная | Описание | Пример |
|------------|----------|--------|
| `VLLM_BASE_URL` | URL OpenAI-совместимого API vLLM | `http://localhost:8000/v1` |
| `VLLM_MODEL` | Имя модели на vLLM | `Qwen/Qwen3-VL-235B-A22B` |
| `VLLM_API_KEY` | Опционально | — |
| `VLLM_TIMEOUT_SECONDS` | Таймаут запроса (сек) | `300` |
| `VLLM_MAX_TOKENS` | Макс. токенов ответа | `8192` |

## Результат распознавания

- **JSON-структура** — массив элементов по страницам: `type` (text, image, table, stamp, signature), `bbox` [x1, y1, x2, y2] в шкале 0–1000, `text`/`content`.
- **Markdown** — собранный из этих элементов текст с разделителями страниц.
- **Веб-интерфейс** — вкладка «Страницы с bbox»: изображение каждой страницы с нарисованными bbox по типам (цвета: текст, таблица, изображение, печать, подпись).

## Локальная разработка без Docker

```bash
python -m venv .venv
.venv\Scripts\activate   # или source .venv/bin/activate
pip install -r requirements.txt
# Установите poppler (pdf2image): на Ubuntu — sudo apt install poppler-utils
cp .env.example .env
# Заполните VLLM_BASE_URL
uvicorn app.main:app --reload --port 8000
```

Откройте http://localhost:8000.
