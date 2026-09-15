"""Model abstractions for semantic database oracles and proxies.

Defines:
- BaseOracle and BaseProxy abstract classes.
- LiteLLMOracle and LiteLLMProxy for LLM-based inference with cost estimation,
  concurrency control, and token logprob scoring.
- CallableOracle and CallableProxy for arbitrary custom/conventional models.
"""

from __future__ import annotations

import asyncio
import inspect
import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import litellm
import yaml
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionSystemMessage,
    ChatCompletionUserMessage,
)

ConfigType = (
    str | Path | dict[str, Any] | list[dict[str, Any]] | litellm.Router | None
)


@dataclass
class CostEstimate:
    """Estimated tokens and cost for model inference."""

    total_items: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost: float
    model_name: str

    def summary(self) -> str:
        return (
            f"--- Cost Estimate ({self.model_name}) ---\n"
            f"Items to process:     {self.total_items:,}\n"
            f"Est. Prompt Tokens:   {self.prompt_tokens:,}\n"
            f"Est. Compl. Tokens:   {self.completion_tokens:,}\n"
            f"Est. Total Tokens:    {self.total_tokens:,}\n"
            f"Est. Cost:            ${self.estimated_cost:,.4f}\n"
            f"----------------------------------------"
        )


class BaseOracle(ABC):
    """Abstract base class for ground-truth oracle models."""

    @abstractmethod
    def predict(
        self, items: list[Any], query: Any
    ) -> tuple[list[bool], list[dict[str, Any]]]:
        """
        Predicts binary boolean labels and per-item cost dictionaries.

        Returns:
            A tuple (labels, costs):
                - labels: list of boolean predictions.
                - costs: list of cost dictionaries, one per item.
        """
        pass

    def estimate_cost(self, items: list[Any], query: Any) -> CostEstimate | None:
        """Estimates token usage and monetary cost for predicting the items."""
        return None


class BaseProxy(ABC):
    """Abstract base class for helper proxy scoring models."""

    @abstractmethod
    def score(
        self, items: list[Any], query: Any
    ) -> tuple[list[float], list[dict[str, Any]]]:
        """
        Computes continuous confidence scores in [0.0, 1.0] and per-item cost dicts.

        Returns:
            A tuple (scores, costs):
                - scores: list of float confidence scores.
                - costs: list of cost dictionaries, one per item.
        """
        pass

    def estimate_cost(self, items: list[Any], query: Any) -> CostEstimate | None:
        """Estimates token usage and monetary cost for scoring the items."""
        return None


def _parse_callable_output(
    res: Any,
    cast_fn: Callable[[Any], Any],
    default_cost: dict[str, Any],
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Parses return value from callable into (typed_values, cost_dicts)."""
    if (
        isinstance(res, tuple)
        and len(res) == 2
        and isinstance(res[1], (list, tuple))
    ):
        values, costs = res
        return [cast_fn(x) for x in values], list(costs)
    values = [cast_fn(x) for x in res]
    return values, [default_cost.copy() for _ in values]


class CallableOracle(BaseOracle):
    """Wraps any user-provided callable as an oracle."""

    def __init__(
        self,
        func: Callable[..., Any],
        default_cost: dict[str, Any] | None = None,
    ):
        self.func = func
        self.default_cost = default_cost.copy() if default_cost is not None else {}

    def predict(
        self, items: list[Any], query: Any
    ) -> tuple[list[bool], list[dict[str, Any]]]:
        return _parse_callable_output(self.func(items, query), bool, self.default_cost)


class CallableProxy(BaseProxy):
    """Wraps any user-provided callable as a proxy scorer."""

    def __init__(
        self,
        func: Callable[..., Any],
        default_cost: dict[str, Any] | None = None,
    ):
        self.func = func
        self.default_cost = default_cost.copy() if default_cost is not None else {}

    def score(
        self, items: list[Any], query: Any
    ) -> tuple[list[float], list[dict[str, Any]]]:
        return _parse_callable_output(
            self.func(items, query), float, self.default_cost
        )


DEFAULT_SCALEDOC_SYSTEM_PROMPT = (
    "You are a helpful assistant. Please answer the question according to the "
    "provided document. Just output the 'Yes' or 'No' only."
)
DEFAULT_SCALEDOC_USER_TEMPLATE = "## Document:\n{doc}\n## Question:\n{query}."


def _default_prompt_formatter(
    item: Any,
    query: Any,
    sys_prompt: str,
    user_tmpl: str,
) -> list[AllMessageValues]:
    doc_str = str(item)
    user_str = user_tmpl.format(doc=doc_str, query=str(query))
    return [
        ChatCompletionSystemMessage(role="system", content=sys_prompt),
        ChatCompletionUserMessage(role="user", content=user_str),
    ]


def _register_custom_pricing(model_list: list[dict[str, Any]]) -> None:
    """Registers each deployment's model_info/pricing fields from config.yaml."""
    cost_map = {}
    for entry in model_list:
        params = entry.get("litellm_params", {}) or {}
        model_info = entry.get("model_info", {}) or {}
        target_model = params.get("model")
        if not target_model:
            continue
        if model_info:
            cost_map[target_model] = model_info
    if cost_map:
        litellm.register_model(cost_map)


def load_litellm_config(
    config: str | Path | dict[str, Any] | list[dict[str, Any]] | litellm.Router,
) -> litellm.Router:
    """
    Initializes a LiteLLM Router from a config path, dict, list, or existing Router.

    Args:
        config: A filepath (str or Path) to a YAML configuration file,
            a dictionary (e.g. {"model_list": [...]}), a list of model deployments,
            or an existing litellm.Router instance.

    Returns:
        Router instance configured with model_list.
    """
    if isinstance(config, litellm.Router):
        return config

    if isinstance(config, (str, Path)):
        path = Path(config)
        if not path.exists():
            raise FileNotFoundError(f"LiteLLM config file not found at '{path}'.")
        with open(path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}
    elif isinstance(config, dict):
        config_data = dict(config)
    elif isinstance(config, list):
        config_data = {"model_list": config}
    else:
        raise TypeError(
            f"Expected config to be str, Path, dict, list, or Router, "
            f"got {type(config).__name__}"
        )

    if "model_list" in config_data:
        model_list = config_data["model_list"]
    elif "model_name" in config_data:
        model_list = [config_data]
    else:
        model_list = []

    _register_custom_pricing(model_list)

    valid_keys = set(inspect.signature(litellm.Router.__init__).parameters.keys())
    router_kwargs: dict[str, Any] = {
        k: v
        for k, v in config_data.items()
        if k in valid_keys and k not in ("self", "model_list")
    }
    return litellm.Router(model_list=model_list, **router_kwargs)


class BaseLiteLLMModel:
    """
    Base class providing common LiteLLM functionality for oracles and proxies.

    Handles message formatting, cost estimation (with regional config.yaml support),
    retry with exponential backoff, and concurrent async batch execution.
    """

    def __init__(
        self,
        model: str,
        system_prompt: str = DEFAULT_SCALEDOC_SYSTEM_PROMPT,
        user_prompt_template: str = DEFAULT_SCALEDOC_USER_TEMPLATE,
        prompt_formatter: Callable[[Any, Any], list[AllMessageValues]] | None = None,
        max_concurrency: int = 10,
        max_retries: int = 5,
        litellm_kwargs: dict[str, Any] | None = None,
        config: ConfigType = None,
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        self.prompt_formatter = prompt_formatter
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        self.litellm_kwargs = dict(litellm_kwargs or {})

        self.router: litellm.Router | None = None
        if config is not None:
            self.router = load_litellm_config(config)

    def format_messages(self, item: Any, query: Any) -> list[AllMessageValues]:
        if self.prompt_formatter is not None:
            return self.prompt_formatter(item, query)
        return _default_prompt_formatter(
            item=item,
            query=query,
            sys_prompt=self.system_prompt,
            user_tmpl=self.user_prompt_template,
        )

    def estimate_cost(self, items: list[Any], query: Any) -> CostEstimate:
        """Estimates total tokens and cost across items."""
        total_items = len(items)
        if total_items == 0:
            return CostEstimate(0, 0, 0, 0, 0.0, self.model)

        # Sample up to 100 items uniformly at random to estimate average prompt tokens
        sample_size = min(total_items, 100)
        sample_items = random.sample(items, sample_size)
        sample_tokens = 0

        for item in sample_items:
            messages = self.format_messages(item, query)
            sample_tokens += litellm.token_counter(model=self.model, messages=messages)

        avg_prompt_tokens = sample_tokens / sample_size
        est_total_prompt_tokens = int(avg_prompt_tokens * total_items)
        est_total_completion_tokens = total_items * 1
        est_total_tokens = est_total_prompt_tokens + est_total_completion_tokens

        # Custom pricing was already registered globally under self.model in
        # load_litellm_config via _register_custom_pricing.
        try:
            prompt_cost, completion_cost = litellm.cost_per_token(
                model=self.model,
                prompt_tokens=est_total_prompt_tokens,
                completion_tokens=est_total_completion_tokens,
            )
            estimated_cost = prompt_cost + completion_cost
        except Exception:
            estimated_cost = 0.0

        return CostEstimate(
            total_items=total_items,
            prompt_tokens=est_total_prompt_tokens,
            completion_tokens=est_total_completion_tokens,
            total_tokens=est_total_tokens,
            estimated_cost=float(estimated_cost),
            model_name=self.model,
        )

    async def _call_litellm(
        self,
        item: Any,
        query: Any,
        semaphore: asyncio.Semaphore,
        **extra_kwargs: Any,
    ) -> Any:
        """Calls litellm.acompletion or router.acompletion with retries and backoff."""
        messages = self.format_messages(item, query)
        kwargs = {
            **self.litellm_kwargs,
            **extra_kwargs,
        }

        delay = 1.0
        for attempt in range(self.max_retries):
            async with semaphore:
                try:
                    if self.router is not None:
                        return await self.router.acompletion(
                            model=self.model,
                            messages=messages,
                            **kwargs
                        )
                    return await litellm.acompletion(
                        model=self.model,
                        messages=messages,
                        **kwargs
                    )
                except Exception as exc:
                    if attempt == self.max_retries - 1:
                        raise RuntimeError(
                            f"LiteLLM call failed after {self.max_retries} "
                            f"attempts: {exc}"
                        ) from exc
                    await asyncio.sleep(delay)
                    delay *= 2.0

    async def _batch_call(
        self,
        items: list[Any],
        query: Any,
        call_fn: Callable[[Any, Any, asyncio.Semaphore], Any],
    ) -> list[Any]:
        semaphore = asyncio.Semaphore(self.max_concurrency)
        tasks = [call_fn(item, query, semaphore) for item in items]
        return await asyncio.gather(*tasks)


def _extract_litellm_cost_metrics(
    response: Any, model: str
) -> dict[str, Any]:
    """Extracts token counts and monetary cost from a LiteLLM ModelResponse."""
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0

    try:
        cost = float(litellm.completion_cost(completion_response=response))
    except Exception:
        try:
            p_cost, c_cost = litellm.cost_per_token(
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            cost = float(p_cost + c_cost)
        except Exception:
            cost = 0.0

    return {
        "monetary": cost,
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
    }


def _extract_litellm_binary_probability(response: Any) -> float:
    """Extracts binary yes/no probability from a LiteLLM logprob response."""
    choice = response.choices[0]
    logprobs_info = getattr(choice, "logprobs", None)

    yes_logprob = None
    no_logprob = None

    if logprobs_info is not None:
        content_logprobs = getattr(logprobs_info, "content", None)
        if content_logprobs and len(content_logprobs) > 0:
            first_token_lp = content_logprobs[0]
            top_logprobs = getattr(first_token_lp, "top_logprobs", [])
            for entry in top_logprobs:
                tok = (
                    entry.token.strip().lower()
                    if hasattr(entry, "token")
                    else ""
                )
                lp = (
                    entry.logprob
                    if hasattr(entry, "logprob")
                    else -99.0
                )
                if tok in ("yes", "y", "true") and yes_logprob is None:
                    yes_logprob = lp
                elif tok in ("no", "n", "false") and no_logprob is None:
                    no_logprob = lp

    if yes_logprob is not None and no_logprob is not None:
        # Softmax over two logits via sigmoid of the log-odds (shift-invariant,
        # avoids underflow when both logprobs are very negative).
        return float(1.0 / (1.0 + math.exp(no_logprob - yes_logprob)))
    if yes_logprob is not None:
        return min(1.0, max(0.0, float(math.exp(yes_logprob))))
    if no_logprob is not None:
        return min(1.0, max(0.0, float(1.0 - math.exp(no_logprob))))

    # Fallback to generated text
    content = (choice.message.content or "").strip().lower()
    if content.startswith("yes"):
        return 1.0
    if content.startswith("no"):
        return 0.0
    return 0.5


class LiteLLMOracle(BaseLiteLLMModel, BaseOracle):
    """
    Oracle model powered by LiteLLM (OpenAI, Azure, Ollama, vLLM, etc.).

    Predicts binary answers (Yes -> True, No -> False) with async concurrency,
    rate limit handling, retry backoff, and upfront cost estimation.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        system_prompt: str = DEFAULT_SCALEDOC_SYSTEM_PROMPT,
        user_prompt_template: str = DEFAULT_SCALEDOC_USER_TEMPLATE,
        prompt_formatter: Callable[[Any, Any], list[AllMessageValues]] | None = None,
        max_concurrency: int = 10,
        max_retries: int = 5,
        litellm_kwargs: dict[str, Any] | None = None,
        config: ConfigType = None,
    ):
        super().__init__(
            model=model,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            prompt_formatter=prompt_formatter,
            max_concurrency=max_concurrency,
            max_retries=max_retries,
            litellm_kwargs=litellm_kwargs,
            config=config,
        )

    async def _call_single(
        self,
        item: Any,
        query: Any,
        semaphore: asyncio.Semaphore,
    ) -> tuple[bool, dict[str, Any]]:
        response = await self._call_litellm(
            item=item,
            query=query,
            semaphore=semaphore,
            max_tokens=1,
        )
        content = response.choices[0].message.content or ""
        clean_str = content.strip().lower()
        label = clean_str.startswith("yes")
        cost = _extract_litellm_cost_metrics(response, self.model)
        return label, cost

    async def apredict(
        self, items: list[Any], query: Any
    ) -> tuple[list[bool], list[dict[str, Any]]]:
        results = await self._batch_call(items, query, self._call_single)
        labels = [r[0] for r in results]
        costs = [r[1] for r in results]
        return labels, costs

    def predict(
        self, items: list[Any], query: Any
    ) -> tuple[list[bool], list[dict[str, Any]]]:
        return asyncio.run(self.apredict(items, query))


class LiteLLMProxy(BaseLiteLLMModel, BaseProxy):
    """
    Proxy scoring model powered by LiteLLM.

    Extracts next-token logprobs for 'Yes' vs 'No' to calculate calibrated
    probabilities P('Yes' | item, query) in [0.0, 1.0].
    """

    def __init__(
        self,
        model: str,
        system_prompt: str = DEFAULT_SCALEDOC_SYSTEM_PROMPT,
        user_prompt_template: str = DEFAULT_SCALEDOC_USER_TEMPLATE,
        prompt_formatter: Callable[[Any, Any], list[AllMessageValues]] | None = None,
        max_concurrency: int = 10,
        max_retries: int = 5,
        litellm_kwargs: dict[str, Any] | None = None,
        config: ConfigType = None,
    ):
        super().__init__(
            model=model,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            prompt_formatter=prompt_formatter,
            max_concurrency=max_concurrency,
            max_retries=max_retries,
            litellm_kwargs=litellm_kwargs,
            config=config,
        )

    async def _call_single(
        self,
        item: Any,
        query: Any,
        semaphore: asyncio.Semaphore,
    ) -> tuple[float, dict[str, Any]]:
        response = await self._call_litellm(
            item=item,
            query=query,
            semaphore=semaphore,
            max_tokens=1,
            logprobs=True,
            top_logprobs=5,
        )
        score = _extract_litellm_binary_probability(response)
        cost = _extract_litellm_cost_metrics(response, self.model)
        return score, cost

    async def ascore(
        self, items: list[Any], query: Any
    ) -> tuple[list[float], list[dict[str, Any]]]:
        results = await self._batch_call(items, query, self._call_single)
        scores = [r[0] for r in results]
        costs = [r[1] for r in results]
        return scores, costs

    def score(
        self, items: list[Any], query: Any
    ) -> tuple[list[float], list[dict[str, Any]]]:
        return asyncio.run(self.ascore(items, query))
