"""Token Factory client with a spending cap.

Every model call in Rootstep goes through `LLM.chat`, so cost is always
counted and a run can never spend more than its budget.
"""
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
LIGHTNING = "nvidia/Nemotron-3_5-Lightning"
SUPER = "nvidia/nemotron-3-super-120b-a12b"
ULTRA = "nvidia/Nemotron-3-Ultra-550b-a55b"
# $ per 1M tokens (input, output), checked 2026-10-04
PRICES = {LIGHTNING: (0.06, 0.24), SUPER: (0.30, 0.90), ULTRA: (1.00, 3.00)}


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Reply:
    content: str            # the visible answer
    reasoning: str | None   # hidden reasoning; recorded, never sent back to the model
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


class LLM:
    def __init__(self, budget_usd: float):
        self.client = AsyncOpenAI(base_url=BASE_URL, api_key=os.environ["NEBIUS_API_KEY"])
        self.budget_usd = budget_usd
        self.spent_usd = 0.0

    async def chat(self, messages: list[dict], model: str = LIGHTNING, temperature: float = 0.7,
                   thinking: bool = False, max_tokens: int = 2048) -> Reply:
        if self.spent_usd >= self.budget_usd:
            raise BudgetExceeded(f"spent ${self.spent_usd:.4f} of ${self.budget_usd:.2f}")
        r = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": thinking}},
        )
        p_in, p_out = PRICES[model]
        cost = (r.usage.prompt_tokens * p_in + r.usage.completion_tokens * p_out) / 1e6
        self.spent_usd += cost
        msg = r.choices[0].message
        return Reply(
            content=(msg.content or "").strip(),
            reasoning=(msg.model_extra or {}).get("reasoning_content"),
            prompt_tokens=r.usage.prompt_tokens,
            completion_tokens=r.usage.completion_tokens,
            cost_usd=cost,
        )
