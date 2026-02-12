# Qwen Bbox OCR

Веб-приложение для распознавания PDF через **vLLM** с vision-моделью (Qwen-VL, Ministral-3 и др.): загрузка PDF → конвертация в изображения → OCR по страницам → JSON-структура документа и Markdown. В интерфейсе отображаются страницы с **bbox-разметкой** (текст, таблицы, изображения, печати, подписи).

## Требования

- **Сервер**: Ubuntu 24.04, CUDA, GPU с достаточным VRAM (например 24 ГБ для Qwen2.5-VL-7B или [Ministral-3-14B](https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512)).
- **vLLM** с vision-моделью запускается **отдельно**. Это приложение — клиент: принимает PDF, конвертирует в картинки и вызывает OpenAI-совместимый API vLLM.

## Архитектура

1. **Сервер vLLM** — поднимается отдельно на том же хосте или в сети. На нём развёрнута любая vision-модель с OpenAI-совместимым API (Qwen2.5-VL, Qwen3-VL, Ministral-3-14B и т.д.).
2. **Сервис qwen-bbox-ocr** — FastAPI: загрузка PDF, конвертация в PNG по страницам, запросы к vLLM, возврат JSON + Markdown + base64 изображений страниц для отображения bbox в браузере.

Подключение к vLLM задаётся через **переменные окружения** (файл `.env`). Модель выбирается переменной **`VLLM_MODEL`**.

## Запуск vLLM отдельно (примеры)

На сервере с GPU (vLLM >= 0.12.0 для Ministral):

### Qwen2.5-VL-7B

```bash
pip install vllm
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --served-model-name Qwen/Qwen2.5-VL-7B-Instruct \
  --host 0.0.0.0 --port 8000
```

Для **Qwen3-VL-235B-A22B** укажите соответствующую модель и при необходимости tensor parallelism.

### Ministral-3-14B-Instruct (Mistral, vision, 24GB VRAM)

[Ministral-3-14B-Instruct-2512](https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512) — vision-модель, помещается в 24 ГБ VRAM в FP8. Рекомендуется temperature ниже 0.1 (в приложении уже 0).

```bash
pip install "vllm>=0.12.0"
vllm serve mistralai/Ministral-3-14B-Instruct-2512 \
  --tokenizer_mode mistral --config_format mistral --load_format mistral \
  --host 0.0.0.0 --port 8000
```

В `.env` укажите:
```env
VLLM_MODEL=mistralai/Ministral-3-14B-Instruct-2512
```

Имя в `VLLM_MODEL` должно совпадать с тем, что возвращает `curl http://localhost:8000/v1/models`.

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

**Когда приложение в Docker, а vLLM на хосте (например в tmux):** из контейнера `localhost` указывает на сам контейнер, а не на хост — запросы до vLLM не доходят. В `.env` обязательно укажите адрес **хоста**:

- **Linux**: `VLLM_BASE_URL=http://172.17.0.1:8000/v1` (шлюз docker0; если не подходит — используйте реальный IP хоста, например `192.168.x.x`).
- **Windows/Mac**: `VLLM_BASE_URL=http://host.docker.internal:8000/v1`.

Убедитесь, что vLLM слушает на `0.0.0.0:8000` (а не только `127.0.0.1`), иначе хост не примет соединения с docker0.

## Переменные окружения (.env)

| Переменная | Описание | Пример |
|------------|----------|--------|
| `VLLM_BASE_URL` | URL OpenAI-совместимого API vLLM | `http://localhost:8000/v1` |
| `VLLM_MODEL` | Имя модели на vLLM (Qwen-VL, Ministral-3-14B и др.) | `Qwen/Qwen2.5-VL-7B-Instruct` или `mistralai/Ministral-3-14B-Instruct-2512` |
| `VLLM_API_KEY` | Опционально | — |
| `VLLM_TIMEOUT_SECONDS` | Таймаут запроса (сек) | `300` |
| `VLLM_MAX_TOKENS` | Макс. токенов ответа | `2048` |
| `PDF_DPI` | DPI при конвертации PDF в картинки (меньше — меньше токенов на изображение) | `150` |

## Преодоление лимита токенов

Если ответ модели обрезается (в логах «ответ обрезан по токенам») или vLLM возвращает 400 про `max_tokens`:

1. **Увеличить контекст на стороне vLLM** — при запуске vLLM задать больший контекст (Qwen2.5-VL поддерживает до 128K, по умолчанию может стоять 4096):
   ```bash
   python -m vllm.entrypoints.openai.api_server \
     --model Qwen/Qwen2.5-VL-7B-Instruct \
     --host 0.0.0.0 --port 8000 \
     --max-model-len 8192
   ```
   Тогда в `.env` можно поднять `VLLM_MAX_TOKENS=2048` или `4096`.

2. **Увеличить `VLLM_MAX_TOKENS` в `.env`** — в пределах, которые разрешает vLLM: `input_tokens + max_tokens <= max_model_len`. Если vLLM запущен с `--max-model-len 8192`, можно ставить 2048–4096.

3. **Уменьшить входные токены** — меньше картинка → меньше токенов на изображение, больше остаётся на ответ. В `.env` задайте `PDF_DPI=100` (или 72). Качество распознавания чуть снизится, зато ответ будет реже обрезаться.

## Устранение неполадок

**Запросы не доходят до vLLM (в логах модели ничего нет, приложение «висит» на «отправка страницы»):** приложение в Docker, vLLM на хосте. В контейнере `localhost` — это сам контейнер. В `.env` задайте адрес хоста: `VLLM_BASE_URL=http://172.17.0.1:8000/v1` (Linux). Запускайте vLLM с `--host 0.0.0.0`, чтобы он принимал подключения с docker0.

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
