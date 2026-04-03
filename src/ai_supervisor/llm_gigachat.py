from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from ai_supervisor.llm_base import LLMClient

logger = logging.getLogger(__name__)

_pool = ThreadPoolExecutor(max_workers=2)


class GigaChatLLMClient(LLMClient):
    """
    Обёртка над официальным SDK GigaChat (синхронный), вызывается в пуле потоков.
    Требуется: pip install gigachat
    """

    def __init__(self, *, credentials: str, model: str | None = None) -> None:
        try:
            from gigachat import GigaChat  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "Для GigaChat установите зависимость: pip install gigachat"
            ) from e
        self._credentials = credentials
        self._model = model
        self._GigaChat = GigaChat

    async def complete(self, *, system: str, user: str) -> str:
        loop = asyncio.get_running_loop()

        def _call() -> str:
            kwargs: dict[str, object] = {
                "credentials": self._credentials,
                "verify_ssl_certs": False,
            }
            if self._model:
                kwargs["model"] = self._model
            with self._GigaChat(**kwargs) as g:
                # GigaChat: объединяем system+user в один запрос (зависит от версии SDK)
                prompt = f"{system}\n\n---\n\n{user}"
                resp = g.chat(prompt)
                return resp.choices[0].message.content

        return await loop.run_in_executor(_pool, _call)
