"""
Usage tracking: tokens, cost, and latency per model, plus a persistent log.

Prices come from the `pricing` section of config.yaml, in USD per 1M tokens.
A model without a price is still tracked; its cost just shows as n/a.
"""
import json
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class ModelStats:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    fully_priced: bool = True
    ttft_sum: float = 0.0
    ttft_count: int = 0


def fmt_cost(cost):
    if cost is None:
        return "n/a"
    return f"${cost:.6f}" if cost < 0.01 else f"${cost:.4f}"


class UsageTracker:
    def __init__(self, pricing=None, log_path=None):
        self.pricing = pricing or {}
        self.log_path = log_path
        self.by_model = {}

    def cost_for(self, model, input_tokens, output_tokens):
        """Cost of one request in USD, or None if the model has no price."""
        price = self.pricing.get(model)
        if not price:
            return None
        return (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000

    def record(self, provider, model, input_tokens, output_tokens, ttft=None, duration=None):
        """Add one request to the session totals and the log. Returns its cost."""
        cost = self.cost_for(model, input_tokens, output_tokens)

        s = self.by_model.setdefault(model, ModelStats())
        s.requests += 1
        s.input_tokens += input_tokens
        s.output_tokens += output_tokens
        if cost is None:
            s.fully_priced = False
        else:
            s.cost += cost
        if ttft is not None:
            s.ttft_sum += ttft
            s.ttft_count += 1

        if self.log_path:
            entry = {
                "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "provider": provider,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost,
                "ttft_s": round(ttft, 3) if ttft is not None else None,
                "duration_s": round(duration, 3) if duration is not None else None,
            }
            try:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + "\n")
            except OSError:
                pass  # logging must never break the chat

        return cost

    @property
    def has_data(self):
        return bool(self.by_model)

    def summary(self):
        """A table of this session's usage, per model and in total."""
        if not self.by_model:
            return "No requests yet this session."

        header = f"{'Model':<26}{'Reqs':>5}{'In':>9}{'Out':>9}{'Cost':>12}{'Avg TTFT':>10}"
        lines = [header, "-" * len(header)]
        total = ModelStats()

        for model, s in self.by_model.items():
            avg = f"{s.ttft_sum / s.ttft_count:.2f}s" if s.ttft_count else "n/a"
            cost = fmt_cost(s.cost) if s.fully_priced else (fmt_cost(s.cost) + "*" if s.cost else "n/a")
            lines.append(f"{model:<26}{s.requests:>5}{s.input_tokens:>9}{s.output_tokens:>9}{cost:>12}{avg:>10}")
            total.requests += s.requests
            total.input_tokens += s.input_tokens
            total.output_tokens += s.output_tokens
            total.cost += s.cost
            total.fully_priced = total.fully_priced and s.fully_priced

        lines.append("-" * len(header))
        cost = fmt_cost(total.cost) + ("" if total.fully_priced else "*")
        lines.append(f"{'TOTAL':<26}{total.requests:>5}{total.input_tokens:>9}{total.output_tokens:>9}{cost:>12}")
        if not total.fully_priced:
            lines.append("* some models have no price in config.yaml, so their cost isn't included")
        return "\n".join(lines)