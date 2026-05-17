"""Scoring functions for LLaVA-Med v1.0 evaluation, ported from the v1.0 reference.

This file is a faithful port of the scoring math from:
  - LLaVA-Med-v1.0/llava/eval/eval_metrics/evaluate_metrics.py
  - LLaVA-Med-v1.0/llava/eval/eval_metrics/glossary.py
  - LLaVA-Med-v1.0/llava/eval/eval_metrics/utils.py
  - LLaVA-Med-v1.0/llava/eval/run_eval.py  (closed-question gate + orchestration)

The math is reproduced byte-for-byte. The only additions are:
  1. A `score_predictions(...)` orchestrator that adapts the harness's
     VQASample-and-prediction shape to v1.0's scoring functions. v1.0's
     own orchestrator expects ground truth in LLaVA conversation format
     ({'conversations': [{'from': 'human', ...}, {'from': 'gpt', ...}]});
     ours uses flat VQASample attributes (.question, .answer, .answer_type).
  2. Case-insensitive handling of answer_type. v1.0 uses 'OPEN'/'CLOSED'
     (uppercase); our loader emits 'open'/'closed' (lowercase). We accept
     both throughout.

KNOWN PUBLISHED-AND-REPRODUCED BEHAVIORS (these are not bugs in this file):
  - Closed scoring is yes/no-substring-gated. Non-yes/no closed questions
    ("which side?", "what modality?") always score 0 regardless of
    prediction. After Bug #3 fix (May 15) moved 21 VQA-RAD questions into
    the closed bucket, ~21 questions are unscorable by this gate.
  - Closed scoring uses `gt in pred` substring check, admitting false
    positives like "no abnormalities" scoring 1 when GT is "yes" -- if
    the predication contains both 'yes' and 'no' anywhere.
  - Open `calculate_appearance_with_normalization` always picks SOME
    candidate via argmax-of-similarities, even when zero candidate words
    appear in the prediction (argmax of all-zeros returns index 0). At
    402 candidates, random-hit rate is ~0.25%.
  - `calculate_exactmatch` returns `(reference_words_in_prediction) /
    (total_words_in_prediction)`. Despite the name, this is
    PREDICTION-precision-like, not reference-recall. The v1.5 harness's
    `open_recall` measured a different quantity; do not cross-compare.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from nltk.translate.bleu_score import sentence_bleu


# =============================================================================
# Section 1: glossary.py — VQA-v2-style answer normalization
# Ported verbatim from LLaVA-Med-v1.0/llava/eval/eval_metrics/glossary.py
# =============================================================================

contractions = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
    "couldnt": "couldn't", "couldn'tve": "couldn't've", "couldnt've": "couldn't've",
    "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't",
    "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't",
    "havent": "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've",
    "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's",
    "Id've": "I'd've", "I'dve": "I'd've", "Im": "I'm", "Ive": "I've",
    "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've",
    "itll": "it'll", "let's": "let's", "maam": "ma'am", "mightnt": "mightn't",
    "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've",
    "oclock": "o'clock", "oughtnt": "oughtn't", "ow's'at": "'ow's'at",
    "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't",
    "shed've": "she'd've", "she'dve": "she'd've", "she's": "she's",
    "shouldve": "should've", "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd", "somebodyd've": "somebody'd've",
    "somebody'dve": "somebody'd've", "somebodyll": "somebody'll",
    "somebodys": "somebody's", "someoned": "someone'd",
    "someoned've": "someone'd've", "someone'dve": "someone'd've",
    "someonell": "someone'll", "someones": "someone's",
    "somethingd": "something'd", "somethingd've": "something'd've",
    "something'dve": "something'd've", "somethingll": "something'll",
    "thats": "that's", "thered": "there'd", "thered've": "there'd've",
    "there'dve": "there'd've", "therere": "there're", "theres": "there's",
    "theyd": "they'd", "theyd've": "they'd've", "they'dve": "they'd've",
    "theyll": "they'll", "theyre": "they're", "theyve": "they've",
    "twas": "'twas", "wasnt": "wasn't", "wed've": "we'd've", "we'dve": "we'd've",
    "weve": "we've", "werent": "weren't", "whatll": "what'll",
    "whatre": "what're", "whats": "what's", "whatve": "what've",
    "whens": "when's", "whered": "where'd", "wheres": "where's",
    "whereve": "where've", "whod": "who'd", "whod've": "who'd've",
    "who'dve": "who'd've", "wholl": "who'll", "whos": "who's",
    "whove": "who've", "whyll": "why'll", "whyre": "why're",
    "whys": "why's", "wont": "won't", "wouldve": "would've",
    "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll",
    "y'allll": "y'all'll", "yall'd've": "y'all'd've", "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've",
    "you'dve": "you'd've", "youll": "you'll", "youre": "you're",
    "youve": "you've",
}

manual_map = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3",
    "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
    "nine": "9", "ten": "10",
}

articles = ["a", "an", "the"]
period_strip = re.compile(r"(?!<=\d)(\.)(?!\d)")
comma_strip = re.compile(r"(\d)(\,)(\d)")
punct = [
    ";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-",
    ">", "<", "@", "`", ",", "?", "!",
]


def normalize_word(token: str) -> str:
    """VQA-v2 normalization: punctuation strip + period strip + lowercase
    + article removal + number-word map + contraction reattachment.

    Ported verbatim from glossary.py. Used on both ground truth and
    prediction before any scoring, to bring them to a common form.
    """
    _token = token
    for p in punct:
        if (p + " " in token or " " + p in token) or (
            re.search(comma_strip, token) is not None
        ):
            _token = _token.replace(p, "")
        else:
            _token = _token.replace(p, " ")
    token = period_strip.sub("", _token, re.UNICODE)

    _token = []
    temp = token.lower().split()
    for word in temp:
        word = manual_map.setdefault(word, word)
        if word not in articles:
            _token.append(word)
    for i, word in enumerate(_token):
        if word in contractions:
            _token[i] = contractions[word]
    token = " ".join(_token)
    token = token.replace(",", "")
    return token


# =============================================================================
# Section 2: utils.py — n-gram tokenization and BLEU primitives
# Ported verbatim from LLaVA-Med-v1.0/llava/eval/eval_metrics/utils.py
# =============================================================================


def split_sentence(sentence: str, n: int) -> Dict[str, int]:
    """Split sentence into n-grams, returning a {ngram: count} dict.

    For n=1 this is just a word-frequency table; for n=2 a bigram table; etc.
    The returned dict is used as a Counter throughout the scoring functions.
    """
    words = defaultdict(int)
    tmp_sentence = sentence
    tmp_sentence = tmp_sentence.lower()
    tmp_sentence = tmp_sentence.strip().split()
    length = len(tmp_sentence)
    for i in range(length - n + 1):
        tmp_words = " ".join(tmp_sentence[i: i + n])
        if tmp_words:
            words[tmp_words] += 1
    return words


def brevity_penalty(candidate: str, references: Sequence[str]) -> float:
    """BLEU's brevity penalty. Penalizes predictions shorter than the
    closest-length reference."""
    c = len(candidate)
    ref_lens = (len(reference) for reference in references)
    r = min(ref_lens, key=lambda ref_len: (abs(ref_len - c), ref_len))
    if c > r:
        return 1
    else:
        return math.exp(1 - r / c)


def modified_precision(candidate: str, references: Sequence[str], n: int) -> float:
    """BLEU's clipped n-gram precision."""
    max_frequency = defaultdict(int)
    min_frequency = defaultdict(int)
    candidate_words = split_sentence(candidate, n)
    for reference in references:
        reference_words = split_sentence(reference, n)
        for word in candidate_words:
            max_frequency[word] = max(max_frequency[word], reference_words[word])
    for word in candidate_words:
        min_frequency[word] = min(max_frequency[word], candidate_words[word])
    if sum(candidate_words.values()) == 0:
        return 0.0
    P = sum(min_frequency.values()) / sum(candidate_words.values())
    return P


# =============================================================================
# Section 3: evaluate_metrics.py — the scoring functions themselves
# Ported verbatim from LLaVA-Med-v1.0/llava/eval/eval_metrics/evaluate_metrics.py
# =============================================================================


def calculate_exactmatch(candidate: str, reference: str) -> float:
    """Returns (reference-words-in-candidate) / (total-words-in-candidate).

    NB: despite the name, this is PREDICTION-precision-like, not
    reference-recall. Long verbose predictions are penalized; short focused
    ones are rewarded. The v1.5 harness's `open_recall` is NOT the same
    quantity and should not be cross-compared.
    """
    candidate = normalize_word(candidate)
    reference = normalize_word(reference)
    candidate_words = split_sentence(candidate, 1)
    reference_words = split_sentence(reference, 1)
    count = 0
    total = 0
    for word in reference_words:
        if word in candidate_words:
            count += 1
    for word in candidate_words:
        total += candidate_words[word]
    if total == 0:
        return 0
    else:
        return count / total


def calculate_f1score(candidate: str, reference: str):
    """Token-level F1 between prediction and reference (after normalization).

    Returns (f1, precision, recall). Empty candidate or reference yields
    (0, 0, 0).
    """
    candidate = normalize_word(candidate)
    reference = normalize_word(reference)
    candidate_words = split_sentence(candidate, 1)
    reference_words = split_sentence(reference, 1)
    word_set = set()
    for word in candidate_words:
        word_set.add(word)
    for word in reference_words:
        word_set.add(word)
    tp = 0
    fp = 0
    fn = 0
    for word in word_set:
        if word in candidate_words and word in reference_words:
            tp += candidate_words[word]
        elif word in candidate_words and word not in reference_words:
            fp += candidate_words[word]
        elif word not in candidate_words and word in reference_words:
            fn += reference_words[word]
    if len(candidate_words) == 0:
        return 0, 0, 0
    elif len(reference_words) == 0:
        return 0, 0, 0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if tp == 0:
            return 0, 0, 0
        else:
            return 2 * precision * recall / (precision + recall), precision, recall


def similarity_candidate_prediction(candidate_answer: str, prediction_words: Dict[str, int]) -> float:
    """Used internally by calculate_appearance_with_normalization. Returns
    (candidate-words-in-prediction) / (total-words-in-candidate)."""
    candidate_answer = split_sentence(candidate_answer, 1)
    count = 0
    total = 0
    for word in prediction_words:
        if word in candidate_answer:
            count += 1
    total = len(candidate_answer)
    if total == 0:
        return 0.0
    else:
        return count / total


def _argmax(lst: List[float]) -> int:
    """Return index of max value. Ties broken by lowest index (Python's
    default list.index behaviour)."""
    return lst.index(max(lst))


def calculate_appearance_with_normalization(
    prediction: str, reference: str, candidate_set: Dict[str, List[str]]
) -> float:
    """v1.0's headline open-question scorer.

    Picks the candidate with highest word-overlap to the prediction, then
    returns 1.0 if that candidate equals the (normalized) reference, else
    0.0. On predictions with zero word overlap to any candidate, argmax
    picks the first candidate; random-hit rate is ~1/N (≈0.25% at N=402).

    Expects candidate_set to be {'0': [str, str, ...]}.
    """
    prediction = normalize_word(prediction)
    reference = normalize_word(reference)
    prediction_words = split_sentence(prediction, 1)
    reference_words = split_sentence(reference, 1)

    # v1.0 wraps the candidate list under the key '0'; preserved for compat.
    candidate_list = candidate_set["0"]

    similarity_list = []
    candidate_answer_normalized_list = []
    for candidate_answer in candidate_list:
        if isinstance(candidate_answer, int):
            candidate_answer = str(candidate_answer)
        candidate_answer = normalize_word(candidate_answer)
        candidate_answer_normalized_list.append(candidate_answer)
        similarity_list.append(
            similarity_candidate_prediction(candidate_answer, prediction_words)
        )

    final_prediction = candidate_answer_normalized_list[_argmax(similarity_list)]

    if final_prediction == reference:
        return 1.0
    else:
        return 0.0


# =============================================================================
# Section 4: harness-shaped orchestrator
# This is NOT in v1.0's reference; it's our adapter from VQASample-and-
# prediction shape to v1.0's scoring functions, matching the per-question
# branching that lives in v1.0's run_eval.py.
# =============================================================================


@dataclass
class ScoreReport:
    """Output of score_predictions. Mirrors v1.0's run_eval.py final table
    plus diagnostic counts."""

    # Headline metrics (matched against the v1.0 paper)
    closed_yes_no_accuracy: float = 0.0
    open_appearance_accuracy: float = 0.0  # the candidate-set scorer; v1.0 paper's "open accuracy"

    # Open-question supplementary metrics (also in v1.0's run_eval.py table)
    open_exact_match: float = 0.0
    open_f1: float = 0.0
    open_precision: float = 0.0
    open_recall: float = 0.0
    open_bleu_score: float = 0.0    # cumulative 4-gram BLEU
    open_bleu_score_1: float = 0.0
    open_bleu_score_2: float = 0.0
    open_bleu_score_3: float = 0.0

    # Diagnostic counts
    num_closed_yes_no: int = 0      # questions where v1.0's gate fired (pred contained 'yes' or 'no')
    num_closed_total: int = 0       # all answer_type==closed questions, including non-yes/no
    num_open: int = 0
    num_total: int = 0

    # Per-question breakdowns, for downstream analysis
    per_question: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "closed_yes_no_accuracy": self.closed_yes_no_accuracy,
            "open_appearance_accuracy": self.open_appearance_accuracy,
            "open_exact_match": self.open_exact_match,
            "open_f1": self.open_f1,
            "open_precision": self.open_precision,
            "open_recall": self.open_recall,
            "open_bleu_score": self.open_bleu_score,
            "open_bleu_score_1": self.open_bleu_score_1,
            "open_bleu_score_2": self.open_bleu_score_2,
            "open_bleu_score_3": self.open_bleu_score_3,
            "num_closed_yes_no": self.num_closed_yes_no,
            "num_closed_total": self.num_closed_total,
            "num_open": self.num_open,
            "num_total": self.num_total,
        }


def _normalize_answer_type(at: str) -> str:
    """Bridge the case mismatch: v1.0 uses 'OPEN'/'CLOSED', our loader emits
    'open'/'closed'. Internally we standardize on lowercase."""
    return str(at).strip().lower()


def score_predictions(
    samples: Sequence[Any],
    predictions: Sequence[Dict[str, Any]],
    candidate_set: Dict[str, List[str]],
) -> ScoreReport:
    """Score a full eval run, using v1.0's per-question-type branching.

    Args:
        samples: iterable of VQASample (must have .answer, .answer_type, .question_id)
        predictions: iterable of dicts with at minimum {"question_id": str, "text": str},
                     in the same order as samples (and with matching question_ids).
        candidate_set: the train_open_answers.json content, shape {"0": [str, ...]}.

    Returns:
        ScoreReport with all metrics + per-question breakdowns.
    """
    if len(samples) != len(predictions):
        raise ValueError(
            f"Sample/prediction count mismatch: {len(samples)} vs {len(predictions)}"
        )

    closed_hits: List[int] = []           # yes/no-gated closed hits (v1.0's `closed_scores`)
    open_appearance_hits: List[float] = []  # candidate-set hits
    open_exact_hits: List[float] = []
    open_f1s: List[float] = []
    open_precisions: List[float] = []
    open_recalls: List[float] = []
    open_bleus: List[float] = []
    open_bleu_1s: List[float] = []
    open_bleu_2s: List[float] = []
    open_bleu_3s: List[float] = []

    num_closed_total = 0
    num_open = 0
    per_question: List[Dict[str, Any]] = []

    for sample, pred in zip(samples, predictions):
        # The two strings v1.0 cares about (lowercase + normalized).
        gt_value = str(sample.answer).lower()
        pred_value = str(pred["text"]).lower()
        gt_value = normalize_word(gt_value)
        pred_value = normalize_word(pred_value)

        # v1.0's run_eval.py asserts gt_ids == pred_ids; we check too,
        # since a misalignment here would silently corrupt every metric.
        sample_qid = getattr(sample, "question_id", None)
        pred_qid = pred.get("question_id")
        if sample_qid is not None and pred_qid is not None and sample_qid != pred_qid:
            raise ValueError(
                f"question_id misalignment: sample={sample_qid!r} pred={pred_qid!r}"
            )

        at = _normalize_answer_type(sample.answer_type)
        record: Dict[str, Any] = {
            "question_id": sample_qid or pred_qid,
            "answer_type": at,
            "gt": gt_value,
            "pred": pred_value,
        }

        if at == "open":
            num_open += 1

            # Open headline: candidate-set argmax-then-compare.
            appearance_hit = calculate_appearance_with_normalization(
                pred_value, gt_value, candidate_set
            )
            open_appearance_hits.append(appearance_hit)
            record["appearance_hit"] = appearance_hit

            # Open supplementary: exact-match precision-like, F1, BLEU.
            exact_hit = calculate_exactmatch(pred_value, gt_value)
            open_exact_hits.append(exact_hit)
            record["exact_match"] = exact_hit

            f1, precision, recall = calculate_f1score(pred_value, gt_value)
            open_f1s.append(f1)
            open_precisions.append(precision)
            open_recalls.append(recall)
            record["f1"] = f1
            record["precision"] = precision
            record["recall"] = recall

            # BLEU: v1.0 uses NLTK's sentence_bleu, with their own custom
            # primitives in utils.py also present (unused at the
            # orchestrator level). We follow v1.0's run_eval.py exactly
            # and use NLTK here.
            gt_split = str(gt_value).split()
            pred_split = str(pred_value).split()
            b = sentence_bleu(references=[gt_split], hypothesis=pred_split)
            b1 = sentence_bleu(references=[gt_split], hypothesis=pred_split,
                               weights=(1, 0, 0, 0))
            b2 = sentence_bleu(references=[gt_split], hypothesis=pred_split,
                               weights=(0, 1, 0, 0))
            b3 = sentence_bleu(references=[gt_split], hypothesis=pred_split,
                               weights=(0, 0, 1, 0))
            open_bleus.append(b)
            open_bleu_1s.append(b1)
            open_bleu_2s.append(b2)
            open_bleu_3s.append(b3)
            record["bleu"] = b
            record["bleu_1"] = b1
            record["bleu_2"] = b2
            record["bleu_3"] = b3

        elif at == "closed":
            num_closed_total += 1

            # v1.0's closed gate: only score if prediction contains 'yes'
            # or 'no'. Otherwise the question scores 0 and is *still
            # counted* in closed_scores (matching v1.0's run_eval.py:
            # `closed_scores['hit'].append(0)` in the else branch).
            if "yes" in pred_value or "no" in pred_value:
                hit = 1 if gt_value in pred_value else 0
            else:
                hit = 0
            closed_hits.append(hit)
            record["closed_hit"] = hit
            record["yes_no_gate_fired"] = ("yes" in pred_value or "no" in pred_value)

        else:
            # Unknown answer_type: log and skip. Better to surface than
            # silently roll into a bucket.
            record["warning"] = f"unknown answer_type: {sample.answer_type!r}"

        per_question.append(record)

    # === Aggregate ============================================================
    report = ScoreReport()
    report.num_open = num_open
    report.num_closed_total = num_closed_total
    report.num_closed_yes_no = sum(
        1 for r in per_question
        if r.get("answer_type") == "closed" and r.get("yes_no_gate_fired")
    )
    report.num_total = num_open + num_closed_total

    if closed_hits:
        report.closed_yes_no_accuracy = sum(closed_hits) / len(closed_hits)
    if open_appearance_hits:
        report.open_appearance_accuracy = sum(open_appearance_hits) / len(open_appearance_hits)
    if open_exact_hits:
        report.open_exact_match = sum(open_exact_hits) / len(open_exact_hits)
    if open_f1s:
        report.open_f1 = sum(open_f1s) / len(open_f1s)
        report.open_precision = sum(open_precisions) / len(open_precisions)
        report.open_recall = sum(open_recalls) / len(open_recalls)
    if open_bleus:
        report.open_bleu_score = sum(open_bleus) / len(open_bleus)
        report.open_bleu_score_1 = sum(open_bleu_1s) / len(open_bleu_1s)
        report.open_bleu_score_2 = sum(open_bleu_2s) / len(open_bleu_2s)
        report.open_bleu_score_3 = sum(open_bleu_3s) / len(open_bleu_3s)

    report.per_question = per_question
    return report