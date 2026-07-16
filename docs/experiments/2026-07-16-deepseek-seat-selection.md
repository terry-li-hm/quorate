# DeepSeek V4 Pro seat selection

Date: 2026-07-16

## Hypothesis

DeepSeek V4 Pro can replace MiMo v2.5 Pro as the sixth council seat without changing the council's vendor or regional balance, while improving route reliability and general reasoning quality.

## External evidence

Artificial Analysis Intelligence Index v4.1 scored DeepSeek V4 Pro at maximum reasoning 44 and MiMo v2.5 Pro at 42. It also measured DeepSeek at 62 output tokens per second with 1.78 seconds to first token, versus MiMo at 54 tokens per second and 2.91 seconds, at the same blended price. The index spans real-world work, banking, coding, knowledge, scientific reasoning, hallucination, and long-context evaluations. Source: [Artificial Analysis comparison](https://artificialanalysis.ai/models/comparisons/deepseek-v4-pro-vs-mimo-v2-5-pro), retrieved 2026-07-16.

DeepSeek's official release describes V4 Pro as a 1.6 trillion parameter mixture-of-experts model with 49 billion active parameters and a one-million-token context window. Source: [DeepSeek V4 release](https://api-docs.deepseek.com/news/news260424), retrieved 2026-07-16.

## Local canary

Quorate's three fixed synthetic canaries ran across all eight production roles with DeepSeek temporarily overriding the sixth seat. DeepSeek was reachable through OpenRouter on all three calls, passed 3 of 3, and averaged 2.07 seconds. Every role passed every canary and the suite retained quorum. The incumbent MiMo result from the earlier same-day production canary was 2 of 3.

An initial attempt was excluded because a local reinstall had replaced Quorate's guarded credential effector, leaving all API-backed seats without credentials. The documented install script restored the wrapper before the valid run. No response text was retained.

After the roster change, the saved production canary remained healthy. DeepSeek again passed 3 of 3 at 2.04 seconds mean latency. Grok was unavailable through both xAI and OpenRouter with HTTP 503 responses, and Gemini passed 2 of 3, but every canary retained its five-seat quorum.

## Decision

Replace MiMo v2.5 Pro with DeepSeek V4 Pro. The council remains six vendors with three Chinese-model seats: Kimi, GLM, and DeepSeek. MiMo remains available through the `CONSILIUM_MODEL_M6` override for future comparisons.

## Limitations

The local suite checks reachability, instruction following, and simple deterministic reasoning rather than substantive council contribution. Artificial Analysis tested maximum reasoning, while Quorate can use a lower provider default. Reconsider the seat if monthly canaries or natural deliberations show weaker reliability or less useful dissent.
