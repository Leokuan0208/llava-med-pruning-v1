"""Scoring orchestrator V2: MCQ-letter for closed, token metrics for open.

Replaces the closed-set scoring in eval/metrics.py's score_predictions() with
MCQ-letter scoring, and removes the candidate-file dependency by dropping the
`appearance_accuracy` open metric. All other open metrics (exact_match, F1,
precision, recall, BLEU) are reused unchanged from eval/metrics.py.

Why a separate file (rather than modifying metrics.py in place): the rewrite
is a load-bearing methodology change; keeping the old scorer intact lets us
compare old-vs-new metrics on the same prediction files for the substring-bug
writeup, and gives us a fallback if the new scorer has its own bugs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from nltk.translate.bleu_score import sentence_bleu

# Reuse the (correct) helpers from the old metrics module. These are not
# buggy; only the closed scoring was.
from eval.metrics import (
    normalize_word,
    calculate_exactmatch,
    calculate_f1score,
)
from eval.mcq import extract_mcq_letter, score_mcq


@dataclass
class ScoreReportV2:
    """Output of score_predictions_v2.

    Closed-set scoring is MCQ-letter based; reported metrics:
      mcq_accuracy:           hits / scorable predictions (extraction succeeded)
      mcq_strict_accuracy:    hits / total yes-no closed (extraction failure = wrong)
      mcq_extraction_rate:    scorable / total yes-no closed

    Diagnostic counts:
      num_closed_yesno:       yes-no closed samples scored under MCQ
      num_closed_unscorable:  non-yes/no closed samples (bucketed, not scored)
      num_extraction_failed:  yes-no closed with extraction failure
    """
    # Closed (MCQ)
    mcq_accuracy: float = 0.0
    mcq_strict_accuracy: float = 0.0
    mcq_extraction_rate: float = 0.0

    # Open (token-level)
    open_exact_match: float = 0.0
    open_f1: float = 0.0
    open_precision: float = 0.0
    open_recall: float = 0.0
    open_bleu_score: float = 0.0
    open_bleu_score_1: float = 0.0
    open_bleu_score_2: float = 0.0
    open_bleu_score_3: float = 0.0

    # Diagnostics
    num_closed_yesno: int = 0
    num_closed_unscorable: int = 0
    num_extraction_failed: int = 0
    num_open: int = 0
    num_total: int = 0

    per_question: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mcq_accuracy": self.mcq_accuracy,
            "mcq_strict_accuracy": self.mcq_strict_accuracy,
            "mcq_extraction_rate": self.mcq_extraction_rate,
            "open_exact_match": self.open_exact_match,
            "open_f1": self.open_f1,
            "open_precision": self.open_precision,
            "open_recall": self.open_recall,
            "open_bleu_score": self.open_bleu_score,
            "open_bleu_score_1": self.open_bleu_score_1,
            "open_bleu_score_2": self.open_bleu_score_2,
            "open_bleu_score_3": self.open_bleu_score_3,
            "num_closed_yesno": self.num_closed_yesno,
            "num_closed_unscorable": self.num_closed_unscorable,
            "num_extraction_failed": self.num_extraction_failed,
            "num_open": self.num_open,
            "num_total": self.num_total,
        }


def score_predictions_v2(
    samples: Sequence[Any],
    predictions: Sequence[Dict[str, Any]],
    mcq_metadata: Dict[str, Dict[str, Any]],
) -> ScoreReportV2:
    """Score a full eval run with V2 methodology.

    Args:
        samples: iterable of VQASample (ORIGINAL samples, not MCQ-mutated).
        predictions: dicts in the same order with {"question_id", "text", ...}.
        mcq_metadata: question_id -> {"correct_letter", "options",
            "formatted_question"} for every yes-no closed sample. Non-yes/no
            closed and open samples are absent.
    """
    if len(samples) != len(predictions):
        raise ValueError(
            f"Sample/prediction count mismatch: {len(samples)} vs {len(predictions)}"
        )

    mcq_hits: List[int] = []
    n_closed_yesno = 0
    n_closed_unscorable = 0
    n_extraction_failed = 0

    open_exact_hits: List[float] = []
    open_f1s: List[float] = []
    open_precisions: List[float] = []
    open_recalls: List[float] = []
    open_bleus: List[float] = []
    open_bleu_1s: List[float] = []
    open_bleu_2s: List[float] = []
    open_bleu_3s: List[float] = []
    n_open = 0

    per_question: List[Dict[str, Any]] = []

    for sample, pred in zip(samples, predictions):
        qid = getattr(sample, "question_id", None) or pred.get("question_id")
        if (sample.question_id != pred.get("question_id")
                and pred.get("question_id") is not None):
            raise ValueError(
                f"question_id misalignment: sample={sample.question_id!r} "
                f"pred={pred.get('question_id')!r}"
            )

        at = str(sample.answer_type).strip().lower()
        record: Dict[str, Any] = {
            "question_id": qid,
            "answer_type": at,
            "gt": str(sample.answer),
            "pred": pred.get("text", ""),
        }

        if at == "closed":
            if qid in mcq_metadata:
                n_closed_yesno += 1
                meta = mcq_metadata[qid]
                correct_letter = meta["correct_letter"]
                extraction = extract_mcq_letter(pred.get("text", ""))
                hit = score_mcq(extraction.letter, correct_letter)

                record["mcq_correct_letter"] = correct_letter
                record["mcq_options"] = list(meta["options"])
                record["mcq_extracted_letter"] = extraction.letter
                record["mcq_extraction_method"] = extraction.method

                if hit is None:
                    n_extraction_failed += 1
                    record["mcq_hit"] = None
                    record["mcq_scorable"] = False
                else:
                    mcq_hits.append(hit)
                    record["mcq_hit"] = hit
                    record["mcq_scorable"] = True
            else:
                n_closed_unscorable += 1
                record["mcq_unscorable_reason"] = "non_yesno_closed"

        elif at == "open":
            n_open += 1
            pred_norm = normalize_word(str(pred.get("text", "")).lower())
            gt_norm = normalize_word(str(sample.answer).lower())

            ex = calculate_exactmatch(pred_norm, gt_norm)
            f1, prec, rec = calculate_f1score(pred_norm, gt_norm)

            gt_split = gt_norm.split()
            pred_split = pred_norm.split()
            b  = sentence_bleu(references=[gt_split], hypothesis=pred_split)
            b1 = sentence_bleu(references=[gt_split], hypothesis=pred_split,
                               weights=(1, 0, 0, 0))
            b2 = sentence_bleu(references=[gt_split], hypothesis=pred_split,
                               weights=(0, 1, 0, 0))
            b3 = sentence_bleu(references=[gt_split], hypothesis=pred_split,
                               weights=(0, 0, 1, 0))

            open_exact_hits.append(ex)
            open_f1s.append(f1)
            open_precisions.append(prec)
            open_recalls.append(rec)
            open_bleus.append(b)
            open_bleu_1s.append(b1)
            open_bleu_2s.append(b2)
            open_bleu_3s.append(b3)

            record["exact_match"] = ex
            record["f1"] = f1
            record["precision"] = prec
            record["recall"] = rec
            record["bleu"] = b
            record["bleu_1"] = b1
            record["bleu_2"] = b2
            record["bleu_3"] = b3

        else:
            record["warning"] = f"unknown answer_type: {sample.answer_type!r}"

        per_question.append(record)

    # === Aggregate =========================================================
    report = ScoreReportV2()
    report.num_closed_yesno = n_closed_yesno
    report.num_closed_unscorable = n_closed_unscorable
    report.num_extraction_failed = n_extraction_failed
    report.num_open = n_open
    report.num_total = n_closed_yesno + n_closed_unscorable + n_open

    if mcq_hits:
        report.mcq_accuracy = sum(mcq_hits) / len(mcq_hits)
    if n_closed_yesno > 0:
        report.mcq_strict_accuracy = sum(mcq_hits) / n_closed_yesno
        report.mcq_extraction_rate = (n_closed_yesno - n_extraction_failed) / n_closed_yesno

    if open_exact_hits:
        report.open_exact_match = sum(open_exact_hits) / len(open_exact_hits)
        report.open_f1 = sum(open_f1s) / len(open_f1s)
        report.open_precision = sum(open_precisions) / len(open_precisions)
        report.open_recall = sum(open_recalls) / len(open_recalls)
        report.open_bleu_score = sum(open_bleus) / len(open_bleus)
        report.open_bleu_score_1 = sum(open_bleu_1s) / len(open_bleu_1s)
        report.open_bleu_score_2 = sum(open_bleu_2s) / len(open_bleu_2s)
        report.open_bleu_score_3 = sum(open_bleu_3s) / len(open_bleu_3s)

    report.per_question = per_question
    return report
