"""
Context management: estimate tokens in the conversation and trim the oldest
turns so the history stays under a token budget.

Token counts are ESTIMATES. Each provider uses its own tokenizer, so we use
OpenAI's tiktoken as a reasonable approximation for every model, falling back
to roughly 4 characters per token if tiktoken isn't available.
"""
try:
    import tiktoken
    _ENCODING = tiktoken.get_encoding("o200k_base")
    TOKENIZER = "tiktoken o200k_base (estimate)"
except Exception:  # not installed, or encoding couldn't be downloaded
    _ENCODING = None
    TOKENIZER = "~4 chars per token (rough estimate)"

# Chat formats add a few hidden tokens per message (role markers, separators)
PER_MESSAGE_OVERHEAD = 4
PER_REQUEST_OVERHEAD = 3


def count_tokens(text):
    if not text:
        return 0
    if _ENCODING is not None:
        return len(_ENCODING.encode(text))
    return max(1, len(text) // 4)


def count_messages(messages):
    """Estimated input tokens for a list of chat messages."""
    return PER_REQUEST_OVERHEAD + sum(
        PER_MESSAGE_OVERHEAD + count_tokens(m["content"]) for m in messages
    )


def trim_history(history, budget):
    """
    Drop the oldest turns until the history fits the budget.

    history[0] is the system message and is always kept, as is the newest
    message. Old user/assistant messages are removed together as a pair so
    the conversation never starts with an orphaned assistant reply.
    Modifies history in place and returns how many messages were dropped.
    """
    dropped = 0
    while count_messages(history) > budget and len(history) > 2:
        history.pop(1)
        dropped += 1
        if len(history) > 2 and history[1]["role"] == "assistant":
            history.pop(1)
            dropped += 1
    return dropped