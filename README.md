# AI Supervisor — мониторинг рабочих чатов MAX

Бот на **Python** использует **[aiomax](https://github.com/dpnspn/aiomax)** (long polling к `platform-api.max.ru`), хранит контекст переписки в **SQLite**, анализирует фрагмент через **YandexGPT** или **GigaChat** и отправляет алерты в общий чат менеджеров и/или в личные сообщения ответственным.

## Требования

- Python **3.11+**
- Токен бота MAX, бот добавлен в групповые чаты и назначен **администратором** (иначе события групп могут не приходить).
- Ключи выбранной LLM (IAM или API-ключ Yandex Cloud для YandexGPT или ключ GigaChat).

## Установка

```bash
cd max_organization
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -e .
```

Опционально: GigaChat — `pip install gigachat`

Скопируйте `.env.example` в `.env` и заполните переменные.

## Запуск

```bash
python -m ai_supervisor
```

Используется long polling через **aiomax** (`SupervisorBot`: таймаут и `marker` совпадают с настройками `LONG_POLL_*`, маркер дополнительно сохраняется в SQLite для перезапусков).

## Переменные окружения

См. `.env.example`. Ключевые:

| Переменная | Описание |
|------------|----------|
| `MAX_ACCESS_TOKEN` | Токен бота MAX |
| `MANAGER_CHAT_ID` | ID чата для алертов |
| `DUTY_USER_IDS` | Список user_id через запятую для личных уведомлений |
| `MONITORED_CHAT_IDS` | Ограничение списка чатов (пусто = все групповые, куда добавлен бот) |
| `LLM_PROVIDER` | `yandex` / `yandexgpt` или `gigachat` |
| `YANDEX_IAM_TOKEN` или `YANDEX_API_KEY` | Авторизация YandexGPT |
| `YANDEX_FOLDER_ID`, `YANDEX_MODEL_URI` | Каталог и URI модели |
| `GIGACHAT_CREDENTIALS` | Ключ авторизации GigaChat |
| `SQLITE_PATH` | Путь к базе контекста и маркера long polling |
| `LONG_POLL_TIMEOUT`, `LONG_POLL_LIMIT` | Параметры `GET /updates` |
| `CONTEXT_WINDOW_MESSAGES` | Сколько последних реплик отдавать в LLM |
| `ANALYSIS_DEBOUNCE_SECONDS` | Задержка перед анализом («пачка» сообщений) |

## systemd

Файл примера: `deploy/ai-supervisor.service`. Скопируйте проект в `/opt/ai-supervisor`, создайте venv, положите `.env`, отредактируйте пользователя `User`/`Group` при необходимости:

```bash
sudo cp deploy/ai-supervisor.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ai-supervisor
```

## Архитектура

- `ai_supervisor.aiomax_bot` — подкласс `aiomax.Bot` (marker в SQLite, timeout/limit).
- `ai_supervisor.supervisor` — обработчик `@on_message`, история, LLM, дебаунс.
- `ai_supervisor.llm_yandex` / `llm_gigachat` — провайдеры моделей.
- `ai_supervisor.adapters` — заготовка под второй мессенджер (Telegram).

## Тесты

```bash
pip install -e ".[dev]"
pytest tests
```

## Юридическое

Расходы на токены LLM и изменения API MAX не входят в поставку кода; при смене контракта API может потребоваться обновление **aiomax** или обёртки.
