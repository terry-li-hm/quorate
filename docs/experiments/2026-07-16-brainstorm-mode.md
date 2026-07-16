# Brainstorm mode validation

Date: 2026-07-16

## Hypothesis

A dedicated divergent pipeline will produce a broader and more usable idea set than the conversational `discuss` preset. Independent lenses should reduce anchoring, one paired cross-pollination round should create combinations without full-group convergence, and a separate curator should remove duplicates without erasing a high-risk wildcard.

## Design

The six council families generate alongside Gemini 3.5 Flash and MiniMax M3. Eight fixed lenses cover first principles, unmet user tension, contrarian inversion, adjacent-industry transfer, constraint removal, recombination, long-horizon change, and cheap experiments. Models generate independently, then each successful model receives one peer's seed response. Claude Fable 5 curates six ideas, one wildcard, and a pattern map, with GPT-5.6 Sol as the output-contract fallback.

MiniMax is isolated to brainstorm mode and production canaries rather than added to the decision council. Artificial Analysis scored MiniMax M3 at 44 on its Intelligence Index and measured competitive speed and price. Source: [MiniMax M3 analysis](https://artificialanalysis.ai/models/minimax-m3), retrieved 2026-07-16.

## Validation

The nine-role production canary was healthy. Every model passed all three fixed prompts; MiniMax was reachable through OpenRouter on 3 of 3 calls and averaged 1.67 seconds.

The first sanitized end-to-end run reached quorum with seven successful generators and seven cross-pollination responses, but Fable returned only a verification caveat rather than the required shortlist. The mode was changed to validate all three required headings and fall back to GPT when the contract is incomplete.

The second sanitized run completed in 219.9 seconds with eight successful generators, eight successful hybrids, and Fable curation containing the shortlist, wildcard, and pattern map. No raw model response is retained in this repository.

## Decision

Ship `quorate brainstorm` as a first-class mode and allow the classifier to select it for requests seeking ideas, concepts, names, possibilities, or opportunities. Keep `discuss` for conversational exploration and keep the six-seat decision council unchanged.

## Limitations

The smoke test verified route health, phase behavior, and output structure, not whether the shortlisted ideas outperform a single-model baseline. The fixed lenses may become predictable across repeated use, and paired cross-pollination explores fewer combinations than an all-to-all round. Evaluate natural brainstorming outcomes before expanding the roster or adding more rounds.
