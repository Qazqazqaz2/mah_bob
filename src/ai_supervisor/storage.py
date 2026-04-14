from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Set, Tuple

from ai_supervisor.config import Settings

logger = logging.getLogger(__name__)

MonitorMode = Literal["all", "list"]
LLMProvider = Literal["yandex", "gigachat"]


@dataclass(frozen=True)
class StoredLine:
    ts: int
    sender_id: Optional[int]
    sender_name: str
    text: str


class SupervisorStorage:
    """SQLite: маркер long polling и история сообщений по чатам."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._lock = threading.Lock()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._migrate_chat_messages_if_needed()
        self._order_col = self._init_chat_messages_order_column()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as c:
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS kv (
                  key TEXT PRIMARY KEY,
                  value TEXT NOT NULL
                )
                """
            )
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  chat_id INTEGER NOT NULL,
                  ts INTEGER NOT NULL,
                  sender_id INTEGER,
                  sender_name TEXT NOT NULL,
                  text TEXT NOT NULL
                )
                """
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_chat_ts ON chat_messages(chat_id, ts DESC)"
            )
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS duty_users (
                  user_id INTEGER PRIMARY KEY
                )
                """
            )
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS monitored_chats (
                  chat_id INTEGER PRIMARY KEY
                )
                """
            )
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS known_chats (
                  chat_id INTEGER PRIMARY KEY,
                  title TEXT,
                  last_seen INTEGER NOT NULL
                )
                """
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_known_last ON known_chats(last_seen DESC)"
            )

    @staticmethod
    def _sql_ident(name: str) -> str:
        return '"' + name.replace('"', '""') + '"'

    def _migrate_chat_messages_if_needed(self) -> None:
        """
        Старые БД: таблица без столбца id или с WITHOUT ROWID — в SQLite нет псевдостолбца rowid.
        Пересоздаём таблицу с id INTEGER PRIMARY KEY AUTOINCREMENT.
        """
        with self._lock, self._connect() as c:
            row = c.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='chat_messages'"
            ).fetchone()
            if not row or not row[0]:
                return
            ddl = row[0]
            info = c.execute("PRAGMA table_info(chat_messages)").fetchall()
            col_names = [x[1] for x in info]
            cset = set(col_names)
            without_rowid = "WITHOUT ROWID" in ddl.upper()
            missing_id = "id" not in cset
            if not without_rowid and not missing_id:
                return

            logger.info(
                "Миграция chat_messages (WITHOUT ROWID=%s, есть id=%s)",
                without_rowid,
                "id" in cset,
            )
            if "chat_id" not in cset or "ts" not in cset:
                logger.error(
                    "chat_messages: нет chat_id/ts, столбцы=%s — миграция пропущена",
                    col_names,
                )
                return

            text_col = next(
                (
                    n
                    for n in (
                        "text",
                        "body",
                        "message",
                        "content",
                        "msg",
                        "plaintext",
                        "txt",
                        "caption",
                    )
                    if n in cset
                ),
                None,
            )
            sender_col = next(
                (n for n in ("sender_name", "author", "sender", "from_name") if n in cset),
                None,
            )
            if not text_col:
                logger.error(
                    "chat_messages: нет столбца текста (text/body/…); есть %s",
                    col_names,
                )
                return
            if not sender_col:
                sender_sel = "''"
            else:
                sender_sel = f"COALESCE({self._sql_ident(sender_col)}, '')"

            text_sel = self._sql_ident(text_col)
            sender_id_expr = (
                self._sql_ident("sender_id")
                if "sender_id" in cset
                else "CAST(NULL AS INTEGER)"
            )

            c.execute("BEGIN IMMEDIATE")
            try:
                c.execute("ALTER TABLE chat_messages RENAME TO _chat_messages_old")
                c.execute(
                    """
                    CREATE TABLE chat_messages (
                      id INTEGER PRIMARY KEY AUTOINCREMENT,
                      chat_id INTEGER NOT NULL,
                      ts INTEGER NOT NULL,
                      sender_id INTEGER,
                      sender_name TEXT NOT NULL,
                      text TEXT NOT NULL
                    )
                    """
                )
                c.execute(
                    f"""
                    INSERT INTO chat_messages(chat_id, ts, sender_id, sender_name, text)
                    SELECT chat_id, ts, {sender_id_expr}, {sender_sel}, {text_sel}
                    FROM _chat_messages_old
                    """
                )
                c.execute("DROP TABLE _chat_messages_old")
                c.execute(
                    "CREATE INDEX IF NOT EXISTS idx_chat_ts ON chat_messages(chat_id, ts DESC)"
                )
                c.commit()
            except Exception:
                c.execute("ROLLBACK")
                logger.exception("Миграция chat_messages не удалась")
                raise

    def _init_chat_messages_order_column(self) -> str:
        """
        У старых таблиц мог не быть столбца id, но есть rowid.
        Без id и без rowid (WITHOUT ROWID) — только миграция/пересоздание БД.
        """
        with self._lock, self._connect() as c:
            exists = c.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='chat_messages'"
            ).fetchone()
            if not exists:
                return "id"
            names_lower = {
                x[1].lower() for x in c.execute("PRAGMA table_info(chat_messages)").fetchall()
            }
            if "id" in names_lower:
                return "id"
            row = c.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='chat_messages'"
            ).fetchone()
            ddl = (row[0] or "") if row else ""
            if "WITHOUT ROWID" in ddl.upper():
                raise RuntimeError(
                    "Таблица chat_messages в режиме WITHOUT ROWID без столбца id — "
                    "SQLite не даёт rowid. Удалите файл БД или выполните миграцию вручную."
                )
            try:
                c.execute("SELECT rowid FROM chat_messages LIMIT 1")
            except sqlite3.OperationalError:
                raise RuntimeError(
                    "Таблица chat_messages без столбца id и без доступного rowid."
                ) from None
            logger.info(
                "chat_messages без столбца id — используем rowid для порядка и trim истории"
            )
            return "rowid"

    def _kv_get(self, c: sqlite3.Connection, key: str) -> Optional[str]:
        row = c.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def _kv_set(self, key: str, value: str) -> None:
        with self._lock, self._connect() as c:
            c.execute(
                "INSERT INTO kv(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )

    def _kv_del(self, key: str) -> None:
        with self._lock, self._connect() as c:
            c.execute("DELETE FROM kv WHERE key=?", (key,))

    def seed_runtime_from_env(self, settings: Settings) -> None:
        """Один раз на новую БД: переносит manager/duty/monitored из .env в таблицы."""
        with self._lock, self._connect() as c:
            if self._kv_get(c, "runtime_seeded"):
                return
            mid = settings.manager_chat_id
            if mid is not None:
                c.execute(
                    "INSERT INTO kv(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    ("manager_chat_id", str(mid)),
                )
            for uid in settings.duty_user_id_list():
                c.execute("INSERT OR IGNORE INTO duty_users(user_id) VALUES(?)", (uid,))
            mon = settings.monitored_chat_id_set()
            if mon:
                c.execute(
                    "INSERT INTO kv(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    ("monitor_mode", "list"),
                )
                for cid in mon:
                    c.execute(
                        "INSERT OR IGNORE INTO monitored_chats(chat_id) VALUES(?)",
                        (cid,),
                    )
            else:
                c.execute(
                    "INSERT INTO kv(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    ("monitor_mode", "all"),
                )
            c.execute(
                "INSERT INTO kv(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                ("llm_provider", settings.llm_provider),
            )
            c.execute(
                "INSERT INTO kv(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                ("analysis_debounce_sec", str(settings.analysis_debounce_seconds)),
            )
            c.execute(
                "INSERT INTO kv(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                ("analysis_min_interval_sec", str(settings.analysis_min_interval_seconds)),
            )
            c.execute(
                "INSERT INTO kv(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                ("analysis_max_wait_sec", str(settings.analysis_max_wait_seconds)),
            )
            c.execute(
                "INSERT INTO kv(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                ("runtime_seeded", "1"),
            )

    def get_manager_chat_id(self) -> Optional[int]:
        with self._lock, self._connect() as c:
            v = self._kv_get(c, "manager_chat_id")
        if not v:
            return None
        try:
            return int(v)
        except ValueError:
            return None

    def set_manager_chat_id(self, chat_id: Optional[int]) -> None:
        if chat_id is None:
            self._kv_del("manager_chat_id")
        else:
            self._kv_set("manager_chat_id", str(chat_id))

    def get_monitor_mode(self) -> MonitorMode:
        with self._lock, self._connect() as c:
            v = self._kv_get(c, "monitor_mode")
        return "list" if v == "list" else "all"

    def set_monitor_mode(self, mode: MonitorMode) -> None:
        self._kv_set("monitor_mode", mode)
        if mode == "all":
            with self._lock, self._connect() as c:
                c.execute("DELETE FROM monitored_chats")

    def ensure_monitor_list_mode(self) -> None:
        """Переключить на режим «только список» (таблицу не очищаем)."""
        if self.get_monitor_mode() != "list":
            self._kv_set("monitor_mode", "list")

    def get_llm_provider(self, default: str = "yandex") -> LLMProvider:
        """Провайдер из SQLite; если ключа нет — default (обычно из .env)."""
        with self._lock, self._connect() as c:
            v = self._kv_get(c, "llm_provider")
        if not v:
            d = default.strip().lower()
            return "gigachat" if d == "gigachat" else "yandex"
        s = v.strip().lower()
        if s in {"yandexgpt", "yandex-gpt"}:
            return "yandex"
        if s == "gigachat":
            return "gigachat"
        if s == "yandex":
            return "yandex"
        d = default.strip().lower()
        return "gigachat" if d == "gigachat" else "yandex"

    def set_llm_provider(self, provider: str) -> None:
        s = provider.strip().lower()
        if s in {"yandexgpt", "yandex-gpt"}:
            s = "yandex"
        if s not in ("yandex", "gigachat"):
            raise ValueError("llm_provider: ожидается yandex или gigachat")
        self._kv_set("llm_provider", s)

    def get_analysis_debounce_seconds(self, default: float) -> float:
        with self._lock, self._connect() as c:
            v = self._kv_get(c, "analysis_debounce_sec")
        if v is None or not str(v).strip():
            return float(default)
        try:
            x = float(v)
            return x if x >= 0 else float(default)
        except ValueError:
            return float(default)

    def set_analysis_debounce_seconds(self, sec: float) -> None:
        self._kv_set("analysis_debounce_sec", str(max(0.0, float(sec))))

    def get_analysis_min_interval_seconds(self, default: float) -> float:
        with self._lock, self._connect() as c:
            v = self._kv_get(c, "analysis_min_interval_sec")
        if v is None or not str(v).strip():
            return float(default)
        try:
            x = float(v)
            return x if x >= 0 else float(default)
        except ValueError:
            return float(default)

    def set_analysis_min_interval_seconds(self, sec: float) -> None:
        self._kv_set("analysis_min_interval_sec", str(max(0.0, float(sec))))

    def get_analysis_max_wait_seconds(self, default: float) -> float:
        with self._lock, self._connect() as c:
            v = self._kv_get(c, "analysis_max_wait_sec")
        if v is None or not str(v).strip():
            return float(default)
        try:
            x = float(v)
            return x if x >= 0 else float(default)
        except ValueError:
            return float(default)

    def set_analysis_max_wait_seconds(self, sec: float) -> None:
        self._kv_set("analysis_max_wait_sec", str(max(0.0, float(sec))))

    def _load_json_map(self, key: str) -> dict[str, object]:
        with self._lock, self._connect() as c:
            raw = self._kv_get(c, key)
        if not raw:
            return {}
        try:
            d = json.loads(raw)
            return d if isinstance(d, dict) else {}
        except json.JSONDecodeError:
            return {}

    def _save_json_map(self, key: str, data: dict[str, object]) -> None:
        self._kv_set(key, json.dumps(data, ensure_ascii=False, separators=(",", ":")))

    def is_dialog_suspended(self, user_id: int) -> bool:
        d = self._load_json_map("dialog_suspended_users")
        return bool(d.get(str(int(user_id))))

    def mark_dialog_suspended(self, user_id: int, suspended: bool = True) -> None:
        d = self._load_json_map("dialog_suspended_users")
        k = str(int(user_id))
        if suspended:
            d[k] = 1
        else:
            d.pop(k, None)
        self._save_json_map("dialog_suspended_users", d)

    def get_chat_analyze_hwm(self, chat_id: int) -> Optional[int]:
        d = self._load_json_map("chat_analyze_hwm")
        k = str(chat_id)
        if k not in d:
            return None
        try:
            return int(d[k])  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

    def set_chat_analyze_hwm(self, chat_id: int, ts: int) -> None:
        d = self._load_json_map("chat_analyze_hwm")
        d[str(chat_id)] = int(ts)
        self._save_json_map("chat_analyze_hwm", d)

    def get_chat_last_analysis_wall(self, chat_id: int) -> Optional[float]:
        d = self._load_json_map("chat_last_analysis_wall")
        k = str(chat_id)
        if k not in d:
            return None
        try:
            return float(d[k])  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

    def set_chat_last_analysis_wall(self, chat_id: int, wall: float) -> None:
        d = self._load_json_map("chat_last_analysis_wall")
        d[str(chat_id)] = float(wall)
        self._save_json_map("chat_last_analysis_wall", d)

    def list_monitored_chat_ids(self) -> list[int]:
        with self._lock, self._connect() as c:
            rows = c.execute(
                "SELECT chat_id FROM monitored_chats ORDER BY chat_id"
            ).fetchall()
        return [int(r[0]) for r in rows]

    def add_monitored_chat(self, chat_id: int) -> None:
        with self._lock, self._connect() as c:
            c.execute(
                "INSERT OR IGNORE INTO monitored_chats(chat_id) VALUES(?)",
                (chat_id,),
            )

    def remove_monitored_chat(self, chat_id: int) -> None:
        with self._lock, self._connect() as c:
            c.execute("DELETE FROM monitored_chats WHERE chat_id=?", (chat_id,))

    def clear_monitored_chats(self) -> None:
        with self._lock, self._connect() as c:
            c.execute("DELETE FROM monitored_chats")

    def monitored_chat_filter(self) -> Optional[Set[int]]:
        """None = все группы; иначе только перечисленные id (режим list)."""
        if self.get_monitor_mode() != "list":
            return None
        s = set(self.list_monitored_chat_ids())
        return s

    def list_duty_users(self) -> list[int]:
        with self._lock, self._connect() as c:
            rows = c.execute("SELECT user_id FROM duty_users ORDER BY user_id").fetchall()
        return [int(r[0]) for r in rows]

    def add_duty_user(self, user_id: int) -> None:
        with self._lock, self._connect() as c:
            c.execute("INSERT OR IGNORE INTO duty_users(user_id) VALUES(?)", (user_id,))

    def remove_duty_user(self, user_id: int) -> None:
        with self._lock, self._connect() as c:
            c.execute("DELETE FROM duty_users WHERE user_id=?", (user_id,))

    def touch_known_chat(self, chat_id: int, title: Optional[str]) -> None:
        ts = int(time.time() * 1000)
        with self._lock, self._connect() as c:
            c.execute(
                """
                INSERT INTO known_chats(chat_id, title, last_seen) VALUES(?,?,?)
                ON CONFLICT(chat_id) DO UPDATE SET
                  title=COALESCE(excluded.title, known_chats.title),
                  last_seen=excluded.last_seen
                """,
                (chat_id, title, ts),
            )

    def list_known_chats(self, limit: int = 12) -> List[Tuple[int, Optional[str]]]:
        with self._lock, self._connect() as c:
            rows = c.execute(
                """
                SELECT chat_id, title FROM known_chats
                ORDER BY last_seen DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [(int(r[0]), r[1]) for r in rows]

    def get_marker(self) -> Optional[int]:
        with self._lock, self._connect() as c:
            row = c.execute("SELECT value FROM kv WHERE key='max_marker'").fetchone()
            if not row:
                return None
            return int(row[0])

    def set_marker(self, marker: Optional[int]) -> None:
        if marker is None:
            return
        with self._lock, self._connect() as c:
            c.execute(
                "INSERT INTO kv(key,value) VALUES('max_marker', ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(marker),),
            )

    def append_message(
        self,
        *,
        chat_id: int,
        ts: int,
        sender_id: Optional[int],
        sender_name: str,
        text: str,
        keep_last: int,
    ) -> None:
        oc = self._order_col
        with self._lock, self._connect() as c:
            c.execute(
                """
                INSERT INTO chat_messages(chat_id, ts, sender_id, sender_name, text)
                VALUES(?,?,?,?,?)
                """,
                (chat_id, ts, sender_id, sender_name, text),
            )
            # удаляем старые, оставляя последние keep_last (id или rowid)
            c.execute(
                f"""
                DELETE FROM chat_messages
                WHERE chat_id=?
                  AND {oc} NOT IN (
                    SELECT {oc} FROM (
                      SELECT {oc} FROM chat_messages WHERE chat_id=?
                      ORDER BY ts DESC, {oc} DESC
                      LIMIT ?
                    )
                  )
                """,
                (chat_id, chat_id, keep_last),
            )

    def recent_context(self, chat_id: int, limit: int) -> list[StoredLine]:
        oc = self._order_col
        with self._lock, self._connect() as c:
            rows = c.execute(
                f"""
                SELECT ts, sender_id, sender_name, text FROM (
                  SELECT ts, sender_id, sender_name, text, {oc}
                  FROM chat_messages
                  WHERE chat_id=?
                  ORDER BY ts DESC, {oc} DESC
                  LIMIT ?
                ) AS recent
                ORDER BY ts ASC, {oc} ASC
                """,
                (chat_id, limit),
            ).fetchall()
        return [StoredLine(ts=r[0], sender_id=r[1], sender_name=r[2], text=r[3]) for r in rows]

    def export_state_snapshot(self) -> dict[str, object]:
        with self._lock, self._connect() as c:
            m = c.execute("SELECT value FROM kv WHERE key='max_marker'").fetchone()
            n = c.execute("SELECT COUNT(*) FROM chat_messages").fetchone()
        return {"max_marker": m[0] if m else None, "messages_rows": n[0] if n else 0}


def health_info(storage: SupervisorStorage) -> str:
    return json.dumps(storage.export_state_snapshot(), ensure_ascii=False)
