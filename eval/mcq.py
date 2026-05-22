"""MCQ (multiple-choice question) formatting + scoring for closed-set evaluation.

Replaces the substring-match closed-set scoring in eval/metrics.py with a
prompt-then-parse approach. Closed yes/no questions are reformulated into
2-option MCQ ("A. Yes / B. No" with randomized letter assignment per sample),
the model is prompted to respond with only a letter, and predictions are
scored by letter equality.

Why this design:
  - Avoids the substring bug in v1.0's reference scorer (e.g. "no" matching
    "normal", "shown", "abnormalities").
  - Avoids the verbosity inflation observed under aggressive pruning, where
    pruned models produced longer, hedging answers that lenient metrics
    rewarded incidentally.
  - Letter randomization (A/B vs B/A per sample) prevents the model from
    exploiting positional bias if it always defaults to "A" under uncertainty.
  - Extraction failure is reported as a separate bucket, not silently counted
    as wrong -- that way we can audit prompt compliance separately from
    answer correctness.

Non-yes/no closed questions (~7.7% of VQA-RAD, ~14.7% of SLAKE, 0% of PathVQA)
are bucketed as `unscorable_by_mcq` and excluded from MCQ accuracy. Building
distractor pools to make them N-option MCQ is a future extension; deliberately
out of scope for this rewrite.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Optional, Tuple


# ============================================================================
# Section 1: MCQ formatting
# ============================================================================


@dataclass(frozen=True)
class MCQFormat:
    """Result of formatting a yes/no question as 2-option MCQ.

    Attributes:
        formatted_question: the prompt text passed to the runner (replaces
            the original question).
        correct_letter: 'A' or 'B' -- the letter that maps to the GT answer
            in this particular formatting.
        options: ordered tuple of option labels, e.g. ('Yes', 'No') if the
            randomization left them in canonical order, or ('No', 'Yes')
            if it flipped them. Preserved for audit.
    """
    formatted_question: str
    correct_letter: str
    options: Tuple[str, str]


# The prompt template. Adopted from HuatuoGPT-Vision's eval.py with a small
# adaptation: option list rendered with explicit newlines for clarity, and
# the closing instruction is phrased to be very directive (we've observed
# LLaVA-Med v1.0 has a strong default toward verbose responses, so the
# instruction needs to be unambiguous).
_MCQ_TEMPLATE = (
    "{question}\n\n"
    "A. {opt_a}\n"
    "B. {opt_b}\n\n"
    "Answer with the letter of the correct option only (A or B)."
)


def format_yesno_as_mcq(question: str, gt_answer: str, sample_seed: int) -> MCQFormat:
    """Reformulate a yes/no question as a 2-option MCQ.

    Args:
        question: the original natural-language question.
        gt_answer: the ground-truth answer, must be 'yes' or 'no' (case-
            insensitive). The caller is responsible for filtering non-yes/no
            samples out before calling this.
        sample_seed: deterministic per-sample seed (e.g. hash of question_id)
            so that A/B assignment is stable across runs.

    Returns:
        MCQFormat with the formatted prompt, correct letter, and option order.

    Raises:
        ValueError: if gt_answer (after normalization) isn't 'yes' or 'no'.
    """
    gt_norm = str(gt_answer).strip().lower()
    if gt_norm not in ("yes", "no"):
        raise ValueError(
            f"format_yesno_as_mcq expects gt_answer in {{'yes','no'}}, got {gt_answer!r}"
        )

    # Coin flip on whether to put Yes in slot A or B. Stable per-sample
    # given the same sample_seed.
    rng = random.Random(sample_seed)
    flip = rng.random() < 0.5
    if flip:
        opt_a, opt_b = "No", "Yes"
    else:
        opt_a, opt_b = "Yes", "No"

    if gt_norm == "yes":
        correct_letter = "A" if opt_a == "Yes" else "B"
    else:
        correct_letter = "A" if opt_a == "No" else "B"

    formatted = _MCQ_TEMPLATE.format(
        question=question.strip(),
        opt_a=opt_a,
        opt_b=opt_b,
    )
    return MCQFormat(
        formatted_question=formatted,
        correct_letter=correct_letter,
        options=(opt_a, opt_b),
    )


# ============================================================================
# Section 2: Letter extraction from model predictions
# ============================================================================


@dataclass(frozen=True)
class MCQExtraction:
    """Result of extracting a letter from a model prediction.

    Attributes:
        letter: the extracted letter ('A'/'B'/'C'/'D'/None).
        method: which extraction tier succeeded -- one of
            'tier1_leading_letter', 'tier2_word_boundary', 'failed'.
    """
    letter: Optional[str]
    method: str


# Tier 2 regex: find the first standalone capital letter A-D, allowing for
# common preambles like "The answer is A", "I choose B", "Option C", "A)",
# "(A)". The \b word-boundary anchors prevent matching letters inside words
# (no false positives on "A" inside "Aortic").
_TIER2_PATTERN = re.compile(r"\b([A-D])\b")


def extract_mcq_letter(prediction_text: str) -> MCQExtraction:
    """Extract the model's chosen letter from a prediction string.

    Two-tier strategy:
      Tier 1: prediction's first non-whitespace character is A/B/C/D, and
        the character after it is non-alphabetic (i.e. it's not the start
        of a word like "Aortic"). Most reliable -- model complied with
        the instruction literally.
      Tier 2: a standalone A/B/C/D word appears anywhere in the prediction.
        Catches "The answer is B" or "I would say A".
      Failure: neither tier matched. Reported separately so we can monitor
        prompt-compliance rates.
    """
    if not prediction_text:
        return MCQExtraction(letter=None, method="failed")

    text = prediction_text.strip()
    if not text:
        return MCQExtraction(letter=None, method="failed")

    # Tier 1: leading letter
    if text[0] in "ABCD":
        if len(text) == 1 or not text[1].isalpha():
            return MCQExtraction(letter=text[0], method="tier1_leading_letter")

    # Tier 2: standalone A/B/C/D anywhere
    match = _TIER2_PATTERN.search(text)
    if match:
        return MCQExtraction(letter=match.group(1), method="tier2_word_boundary")

    return MCQExtraction(letter=None, method="failed")


# ============================================================================
# Section 3: Scoring a single prediction
# ============================================================================


def score_mcq(extracted_letter: Optional[str], correct_letter: str) -> Optional[int]:
    """Score a single MCQ prediction.

    Returns:
        1 if extracted_letter == correct_letter
        0 if extracted_letter is a different letter
        None if extraction failed -- caller should bucket these separately
            rather than counting as 0.
    """
    if extracted_letter is None:
        return None
    return 1 if extracted_letter == correct_letter else 0
