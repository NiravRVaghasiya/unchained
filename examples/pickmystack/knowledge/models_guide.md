# LLM Model Guide

A practical guide to choosing a language model. Prices are approximate USD per
1 million tokens and change often; treat them as relative guidance, not quotes.

## OpenAI

### GPT-4o
- Input ~$2.50 / 1M tokens, output ~$10.00 / 1M tokens.
- Strong general reasoning, tool calling, vision, and JSON mode.
- Best for: production assistants needing high quality and reliable tool use.

### GPT-4o mini
- Input ~$0.15 / 1M tokens, output ~$0.60 / 1M tokens.
- Very cheap, fast, good enough for most tool-using agents.
- Best for: cost-sensitive production workloads and high request volume.

## Anthropic

### Claude 3.5 Sonnet
- Input ~$3.00 / 1M tokens, output ~$15.00 / 1M tokens.
- Excellent at coding, long-context reasoning, and careful instruction following.
- Best for: coding agents, analysis, and long-document tasks.

### Claude 3.5 Haiku
- Input ~$0.80 / 1M tokens, output ~$4.00 / 1M tokens.
- Fast and affordable while retaining good quality.
- Best for: latency-sensitive assistants at moderate cost.

## Open weights (self-host with Ollama, vLLM, etc.)

### Llama 3.1 (8B / 70B)
- No per-token API cost when self-hosted; you pay for compute only.
- 8B runs on a single consumer GPU; 70B needs serious hardware or quantization.
- Best for: privacy-sensitive workloads, offline use, predictable flat cost.

### Mistral / Mixtral
- Efficient open-weight models; Mixtral uses a mixture-of-experts design.
- Best for: strong quality per dollar when self-hosting.

### Qwen 2.5
- Competitive open-weight family with strong coding and multilingual ability.
- Best for: multilingual and coding use cases on self-hosted infrastructure.

## Choosing a model

- Tight budget + privacy: self-host Llama 3.1 8B or Mistral via Ollama (no token fees).
- Best quality, cost is secondary: GPT-4o or Claude 3.5 Sonnet.
- High volume, cost matters: GPT-4o mini or Claude 3.5 Haiku.
- Coding-heavy agents: Claude 3.5 Sonnet or Qwen 2.5 Coder.
- Prototyping: start with a cheap or local model, upgrade only if quality falls short.

## Rules of thumb

- Output tokens usually cost several times more than input tokens; watch verbose replies.
- Retrieval (RAG) lets a smaller, cheaper model perform like a larger one on niche data.
- Caching, short system prompts, and tight max_tokens all reduce spend meaningfully.
