import asyncio
import threading
import weakref
from openai import AsyncOpenAI
from typing import Optional


class OpenAIClientManager:
    """OpenAI APIクライアントの安全な管理"""

    _clients = {}
    _lock = threading.Lock()

    @classmethod
    def get_client(cls, api_key: str, base_url: Optional[str] = None) -> AsyncOpenAI:
        client_key = f"{api_key}:{base_url or 'default'}"
        if client_key not in cls._clients:
            with cls._lock:
                if client_key not in cls._clients:
                    client_kwargs = {"api_key": api_key}
                    if base_url:
                        client_kwargs["base_url"] = base_url
                    client = AsyncOpenAI(**client_kwargs) # type: ignore
                    cls._clients[client_key] = client

                    def cleanup():
                        try:
                            if hasattr(client, "close"):
                                asyncio.create_task(client.close())
                        except Exception:
                            pass

                    weakref.finalize(client, cleanup)
        return cls._clients[client_key]

    @classmethod
    async def close_all(cls):
        for client in cls._clients.values():
            try:
                if hasattr(client, "close"):
                    await client.close()
            except Exception:
                pass
        cls._clients.clear()


async def openai_complete_func(
    prompt: str,
    system_prompt: str = "",
    model_name: str = "gpt-4.1",
    api_key: str = "",
    base_url: Optional[str] = None,
    **kwargs,
):
    try:
        client = OpenAIClientManager.get_client(api_key, base_url)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", 8192),
            temperature=kwargs.get("temperature", 0.1),
            top_p=kwargs.get("top_p", 0.9),
        )
        return response.choices[0].message.content if response.choices else ""
    except Exception as e:
        print(f"OpenAI API呼び出しエラー: {e}")
        return ""
