"""Async OpenAI client with structured outputs and usage tracking.

Follows the pattern in adaptive_policy's `apaf.extraction.llm` (Pydantic
response models via structured outputs, exponential backoff) with three
changes:

- async, because a turn issues several independent calls that should overlap;
- one shared client rather than one per call;
- token accounting, since the demo runs on a shared key with finite credits.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import TypeVar

from dotenv import load_dotenv
from openai import APIError, AsyncOpenAI, BadRequestError, RateLimitError
from pydantic import BaseModel

logger = logging.getLogger(__name__)

load_dotenv()

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.6-terra")
#: Optional separate model for the neutral calls (routing, placement). Unset
#: means the persona's model does those jobs too -- with persona-free prompts.
CLASSIFIER_MODEL = os.environ.get("CLASSIFIER_MODEL") or None


@dataclass
class Usage:
    """Cumulative token usage. Cost is only reported if a price is configured."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    #: USD per million tokens, (prompt, completion). Set per model by the caller.
    price_per_mtok: tuple[float, float] | None = None

    def add(self, prompt: int, completion: int) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.calls += 1

    @property
    def estimated_cost_usd(self) -> float | None:
        if self.price_per_mtok is None:
            return None
        p_rate, c_rate = self.price_per_mtok
        return (
            self.prompt_tokens * p_rate + self.completion_tokens * c_rate
        ) / 1_000_000

    def summary(self) -> str:
        base = (
            f"{self.calls} calls, {self.prompt_tokens:,} prompt "
            f"+ {self.completion_tokens:,} completion tokens"
        )
        cost = self.estimated_cost_usd
        return base if cost is None else f"{base}, ~${cost:.4f}"


class MissingAPIKey(RuntimeError):
    pass


class LLMClient:
    """Thin wrapper around AsyncOpenAI returning validated Pydantic models."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.7,
        max_retries: int = 3,
        price_per_mtok: tuple[float, float] | None = None,
    ):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise MissingAPIKey(
                "OPENAI_API_KEY is not set. Put it in a .env file at the project "
                "root, or export it in your shell."
            )
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.usage = Usage(price_per_mtok=price_per_mtok)
        #: Some models accept only the default temperature and 400 on any
        #: explicit value. Detected on first use, then the parameter is dropped.
        self._omit_temperature = False

    async def parse(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        temperature: float | None = None,
    ) -> T:
        """Call the model and return a validated instance of `response_model`."""
        last_error: Exception | None = None
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        attempt = 0
        while attempt < self.max_retries:
            kwargs = {}
            if not self._omit_temperature:
                kwargs["temperature"] = (
                    self.temperature if temperature is None else temperature
                )
            try:
                completion = await self.client.chat.completions.parse(
                    model=self.model,
                    messages=messages,
                    response_format=response_model,
                    **kwargs,
                )
                if completion.usage:
                    self.usage.add(
                        completion.usage.prompt_tokens,
                        completion.usage.completion_tokens,
                    )

                parsed = completion.choices[0].message.parsed
                if parsed is None:
                    raise ValueError("model returned no parseable content")
                return parsed

            except BadRequestError as exc:
                # Probing for temperature support is not a failed attempt, so it
                # does not consume a retry.
                if "temperature" in str(exc) and not self._omit_temperature:
                    logger.info(
                        "%s rejects an explicit temperature; dropping the parameter.",
                        self.model,
                    )
                    self._omit_temperature = True
                    continue
                raise  # any other 400 is a bug in our request, not worth retrying

            except (RateLimitError, APIError, ValueError) as exc:
                last_error = exc
                attempt += 1
                if attempt >= self.max_retries:
                    break
                wait = 2 ** (attempt - 1)
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s. Retrying in %ds.",
                    attempt,
                    self.max_retries,
                    exc,
                    wait,
                )
                await asyncio.sleep(wait)

        raise RuntimeError(
            f"LLM call failed after {self.max_retries} attempts"
        ) from last_error

    async def gather(self, *coros):
        """Run independent calls concurrently.

        A turn typically issues a handful of these -- score a new node, generate
        a rebuttal, re-score an affected sibling -- and they should overlap.
        """
        return await asyncio.gather(*coros)
