# Judge role selection: Fable, Gemini, and Opus

Date: 2026-07-16

Status: adopted in Quorate 0.4.0

Decision commit: `f0a4cc6`

## Question

Should Quorate retain Gemini 3.5 Flash as judge and Claude Opus 4.8 as critic, or promote Claude Fable 5 to judge, move Opus into the council, and use Gemini as the advisory critic?

The judge determines the final recommendation. The critic can expose weaknesses but cannot revise that recommendation, so a quality improvement in the judge role has greater leverage than the same improvement in the critic role.

## Prior evidence

The broad intelligence screen observed on Artificial Analysis was Fable 5 at 60, Opus 4.8 at 56, and Gemini 3.5 Flash at 50. These scores were treated as screening evidence, not as proof of role fitness. Source: [Artificial Analysis model comparison](https://artificialanalysis.ai/models), retrieved 2026-07-16.

A sanitized `council --deep` comparison reached quorum with one council seat missing. Gemini, then the preferred judge, failed and GPT-5.6 Sol produced the synthesis through the configured fallback. It advised retaining the existing architecture until a task-specific comparison was available, while identifying the proposed Fable architecture as the expected successor because judge quality has the greatest downstream leverage. The critique noted that a small benchmark would be imprecise and that model correlations and self-preference required caution.

## Paired synthetic pilot

Twenty hand-built, labeled decision cases tested deterministic reasoning across operational risk, privacy, capacity, statistics, contracts, reliability, and expected value. Fable and Gemini received the same evidence, judgment instructions, high reasoning effort, 1,024 output-token limit, and required final decision marker. Four cases ran concurrently. Inputs were synthetic and contained no private data.

Fable returned the correct labeled decision and completed the response contract in 20 of 20 cases. Its mean call latency was 19.77 seconds. Gemini completed the contract correctly in 9 of 20 cases, with a mean observed latency of 16.62 seconds. Eight Gemini responses were truncated before the final decision marker, although the visible reasoning was substantively correct where enough text remained, and three returned HTTP 403 route failures. Every Gemini response that completed the contract was correct.

The result therefore measures usable judgment, which combines reasoning, completion, instruction following, and route reliability. It does not establish that Gemini's completed reasoning is inaccurate. Concurrency may also overstate Gemini route failures relative to Quorate's single serial judge call. No raw model responses were retained in the repository.

## Production canary

After the role change, `quorate benchmark` ran three fixed synthetic canaries across all eight primary production roles. The suite was healthy and every canary retained its five-seat quorum. Fable, GPT, Opus, Grok, Kimi, and GLM passed 3 of 3. MiMo passed 2 of 3. Gemini passed 1 of 3 because two calls exhausted its preferred and fallback routes. The aggregate snapshot is stored outside the repository at `~/.local/state/quorate/benchmarks/2026-07-16.json`; it contains no response text.

## Decision

Adopt the proposed architecture. Claude Fable 5 is the judge, with GPT-5.6 Sol as fallback. Claude Opus 4.8 replaces Fable on the six-seat council. Gemini 3.5 Flash becomes the advisory critic, with Opus as fallback. The benchmark now includes the critic route so monthly health checks cover every primary production role.

This decision rests on the paired pilot's 20 of 20 usable Fable judgments, the judge role's control over the final answer, and the healthy post-change canary. Reconsider it when task-specific evidence or repeated canaries no longer support Fable's reliability advantage.

## Limitations

The pilot had only 20 cases, used synthetic rather than natural council transcripts, and did not measure bias toward a model's own council contribution. The deep council comparison missed one seat and used the GPT fallback as judge. The study supports the role assignment but does not estimate a precise general-purpose quality difference between the models.
