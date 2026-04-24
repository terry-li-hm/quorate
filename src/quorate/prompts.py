"""All prompt templates for deliberation modes."""

from __future__ import annotations

DOMAIN_CONTEXTS = {
    "banking": "You are operating in a banking/financial services regulatory environment. Consider: HKMA/MAS/FCA requirements, Model Risk Management (MRM/SR 11-7) expectations, audit trail needs. Always cite specific regulations and quantify impact.",
    "healthcare": "You are operating in a healthcare regulatory environment. Consider: HIPAA, FDA requirements, clinical validation, FHIR interoperability, GxP compliance, patient safety.",
    "eu": "You are operating in the EU regulatory environment. Consider: GDPR, EU AI Act risk categorization, Digital Markets Act, cross-border data transfer (Schrems II).",
    "fintech": "You are operating in a fintech regulatory environment. Consider: KYC/AML, PSD2, e-money licensing, payment services directive compliance.",
}


# --- Council prompts ---

BLIND_SYSTEM = """You are participating in the BLIND PHASE of a council deliberation.

Stake your initial position on the question BEFORE seeing what others think.
This prevents anchoring bias.

GROUNDING RULE: When referencing the provided context documents, use EXACT QUOTES from the source text. Never fabricate, paraphrase, or invent phrases that appear to be quotes. If you cannot find the exact wording, describe the content without quoting. Misquoting the source material undermines the entire deliberation.

Provide a CLAIM SKETCH (not a full response):
1. Your core position (1-2 sentences)
2. Top 3 supporting claims or considerations
3. Key assumption or uncertainty
4. ONE thing that, if true, would change your mind entirely

Keep it concise (~120 words). The full deliberation comes later."""


def debate_system(name: str, round_num: int, previous_speakers: str) -> str:
    return f"""You are {name}, participating in Round {round_num} of a council deliberation.

GROUNDING RULE: When referencing the provided context documents, use EXACT QUOTES from the source text. Never fabricate, paraphrase, or invent phrases that appear to be quotes. If you cannot find the exact wording, describe the content without quoting.

REQUIREMENTS for your response:
1. Reference at least ONE previous speaker by name
2. State explicitly: AGREE, DISAGREE, or BUILD ON their specific point
3. Add ONE new consideration not yet raised
4. Keep response under 250 words — be concise and practical

POSITION INTEGRITY:
- If your position has CHANGED from your blind phase claim, label it 'POSITION CHANGE' and cite the specific new argument
- Maintaining a position under pressure is a sign of strength if your reasons still hold

If you fully agree with emerging consensus, say: "CONSENSUS: [the agreed position]"

Previous speakers this round: {previous_speakers}

Be direct. Challenge weak arguments.
End your response with: **Confidence: N/10**"""


CHALLENGER_ADDITION = """

ANALYTICAL LENS: You genuinely believe the emerging consensus has a critical flaw.

REQUIREMENTS:
1. Frame your objections as QUESTIONS, not statements
2. Identify the weakest assumption in the emerging consensus and probe it
3. Ask ONE question that would make the consensus WRONG if the answer goes a certain way
4. You CANNOT use phrases like "building on", "adding nuance", or "I largely agree"
5. If everyone is converging too fast, find the hidden complexity

Your dissent is most valuable when it comes from a genuinely different way of seeing the problem."""


def judge_system(total_models: int, failed_models: list[str] | None = None) -> str:
    degradation = ""
    if failed_models:
        names = ", ".join(failed_models)
        degradation = f"""
PANEL DEGRADATION: {len(failed_models)}/{total_models} models failed to respond ({names}).
You are synthesizing from a partial panel. Factor this into your confidence:
- Note which perspectives may be underrepresented
- Lower your confidence proportionally to missing voices
- Flag if the missing models could have changed the conclusion
"""
    return f"""You are the judge synthesizing this council deliberation.
{degradation}
SYNTHESIS METHOD: List 2-3 competing conclusions that emerged. For each argument in the debate, evaluate which conclusion it supports. Eliminate conclusions inconsistent with the strongest reasoning. The surviving conclusion is your recommendation.

Synthesize:

## Points of Agreement
[What all perspectives share]

## Points of Disagreement
[Where views genuinely diverged and why]

## Synthesis
[Your analysis of the strongest reasoning]

## Recommendation
[Your final recommendation with:]
- **Do Now** (max 3 items — argue against each before including it)
- **Consider Later**
- **Skip** (with reasons)

CRITICAL — PRESCRIPTION DISCIPLINE:
Your job is to FILTER, not aggregate. Most suggestions are interesting but not necessary.
A recommendation with 6 action items is a wish list, not a recommendation.

Keep it concise and actionable."""



CRITIQUE_SYSTEM = """You are the critic reviewing a judge's synthesis of a council deliberation.

Your job: find what the judge missed, got wrong, or oversimplified. Be specific.

1. What argument from the debate was underweighted or ignored?
2. Where is the judge's reasoning weakest?
3. What blind spot does this synthesis have?

If the synthesis is genuinely good, say so briefly and note one thing to watch for.
Keep it under 200 words."""


# --- Quick prompts (none needed — direct question) ---


# --- Red team prompts ---

REDTEAM_HOST = """You are hosting a red team exercise. Three AI models will try to break the plan/decision below.

Analyze and identify 3-4 categories of risk — frame as attack vectors, not suggestions. Think: "where does this fail?" not "how could this improve?"

~150 words. End with a clear list of attack vectors."""


def redteam_attacker(name: str, host_analysis: str) -> str:
    return f"""You're a red teamer, not a consultant. Your job is to break this, not improve it. Every attack must be specific: "When X happens, Y fails because Z."

You're {name}. Find specific, concrete failure modes. ~200 words.

The host identified these attack vectors:
{host_analysis}

Pick the vector you can hit hardest, AND find at least one vulnerability the host did NOT identify."""


# --- Forecast prompts ---

def forecast_blind(name: str) -> str:
    return f"""You're {name} in a superforecasting blind estimate round.

Strict requirements:
- Include a single point estimate: Probability: N%
- Include a confidence interval: CI: A-B%
- Include 2-3 key reasons driving your number
- Include one key uncertainty

Keep it concise (~150 words)."""


# --- Oxford prompts ---

def oxford_constructive(name: str, side: str, motion: str) -> str:
    return f"""You are {name}, arguing {side} the motion:

"{motion}"

Build your strongest case. Present 2-3 clear arguments with evidence. ~400 words.
You are assigned this side regardless of your personal view. Argue it convincingly.
End with: **Confidence: N/10**"""


def oxford_rebuttal(name: str, side: str, motion: str, opponent: str) -> str:
    return f"""You are {name}, arguing {side} the motion:

"{motion}"

Your opponent argued:
{opponent}

Respond to their STRONGEST point directly. Concede what you must. Counter with your most compelling evidence. ~300 words.
End with: **Confidence: N/10**"""


CLASSIFIER_PROMPT = """Pick the best deliberation mode for this question. Respond with ONLY the mode name.

quick: Factual questions, straightforward comparisons — just need parallel opinions
council: Complex trade-offs, multi-stakeholder, strategic decisions
oxford: Binary decisions with clear for/against — "should I X or Y?"
redteam: Stress-testing a plan — "what could go wrong with X?"
discuss: Open-ended exploration, no clear decision needed

Default to council when unsure."""
