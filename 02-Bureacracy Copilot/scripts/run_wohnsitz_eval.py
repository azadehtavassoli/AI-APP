"""Run multilingual evaluation for Wohnsitzanmeldung RAG behavior.

This script performs a deterministic evaluation flow:
1. Clear current knowledge base.
2. Ingest one official webpage and one provided PDF.
3. Run a multilingual question set through `/api/query`.
4. Score answers with a lightweight rubric (correctness/citations/language).
5. Write machine-readable and human-readable reports.

The PDF used for ingestion is intentionally not hard-coded to avoid
user-specific filesystem paths. Provide it via `--pdf-path`.
"""

from __future__ import annotations

import argparse
import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import requests


BASE_URL = "http://127.0.0.1:8000"
WEB_SOURCE = "https://service.berlin.de/dienstleistung/120686/de_plain/"
ROOT = Path(__file__).resolve().parents[1]
REPORT_MD = ROOT / "docs" / "EVALUATION_WOHNSITZ.md"
REPORT_JSON = ROOT / "docs" / "evaluation_wohnsitz_results.json"


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed CLI args.
    """
    parser = argparse.ArgumentParser(
        description="Run a small deterministic evaluation against the classic /api/query RAG path.",
    )
    parser.add_argument(
        "--base-url",
        default=BASE_URL,
        help="Backend base URL (default: http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--web-source",
        default=WEB_SOURCE,
        help="Official webpage URL to ingest before evaluation.",
    )
    parser.add_argument(
        "--pdf-path",
        required=True,
        help="Path to the PDF file to ingest (required).",
    )
    parser.add_argument(
        "--report-md",
        default=str(REPORT_MD),
        help="Output markdown report path.",
    )
    parser.add_argument(
        "--report-json",
        default=str(REPORT_JSON),
        help="Output JSON report path.",
    )

    return parser.parse_args()


@dataclass
class EvalQuestion:
    """Represent one evaluation question with expected keyword groups.

    Each keyword group acts as an OR condition; one hit in each group increases
    correctness confidence.
    """

    question_id: str
    language_label: str
    question_text: str
    response_language: str
    expected_groups: List[List[str]]


def _post(endpoint: str, payload: Dict[str, Any], timeout: int = 120) -> Dict[str, Any]:
    """Send POST request and return parsed JSON payload.

    Args:
        endpoint: Relative API path.
        payload: JSON body.
        timeout: Request timeout in seconds.

    Returns:
        Parsed JSON response payload.

    Raises:
        RuntimeError: If response status is not successful.
    """
    response = requests.post(f"{BASE_URL}{endpoint}", json=payload, timeout=timeout)
    if not response.ok:
        raise RuntimeError(f"POST {endpoint} failed: {response.status_code} {response.text}")
    return response.json()


def _ingest_sources(pdf_path: Path, web_source: str) -> Dict[str, Any]:
    """Clear existing data and ingest website + PDF sources.

    Args:
        pdf_path: Path to the PDF file to ingest.
        web_source: Official webpage URL to ingest.

    Returns:
        Dictionary summarizing ingest calls.
    """
    clear_payload = _post("/api/clear", {})

    web_payload = _post("/api/ingest/url", {"url": web_source}, timeout=180)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pdf_b64 = base64.b64encode(pdf_path.read_bytes()).decode("utf-8")
    pdf_payload = _post(
        "/api/ingest/pdf",
        {
            "pdf_base64": pdf_b64,
            "filename": pdf_path.name,
        },
        timeout=180,
    )

    return {
        "clear": clear_payload,
        "web_ingest": web_payload,
        "pdf_ingest": pdf_payload,
    }


def _build_questions() -> List[EvalQuestion]:
    """Build multilingual evaluation set based on user-provided checklist.

    Returns:
        List of evaluation questions.
    """
    return [
        EvalQuestion(
            "Q1-EN",
            "EN",
            "How many days do I have to register after moving?",
            "en",
            [["14", "fourteen"], ["day", "days"]],
        ),
        EvalQuestion(
            "Q1-DE",
            "DE",
            "Wie viele Tage habe ich nach dem Umzug für die Anmeldung?",
            "de",
            [["14"], ["tag", "tage"]],
        ),
        EvalQuestion(
            "Q1-FA",
            "FA",
            "بعد از اسباب‌کشی چند روز وقت دارم آدرس را ثبت کنم؟",
            "fa",
            [["14", "۱۴"], ["روز"]],
        ),
        EvalQuestion(
            "Q2-EN",
            "EN",
            "What do I need to register online (BundID / AusweisApp / age)?",
            "en",
            [["bundid"], ["ausweisapp", "online-ausweisfunktion", "eid"], ["18", "age"]],
        ),
        EvalQuestion(
            "Q2-DE",
            "DE",
            "Was brauche ich für die Online-Anmeldung (BundID / AusweisApp / Alter)?",
            "de",
            [["bundid"], ["ausweisapp", "online-ausweisfunktion", "eid"], ["18", "alter"]],
        ),
        EvalQuestion(
            "Q3-DE",
            "DE",
            "Was muss in der Einzugsbestätigung vom Wohnungsgeber stehen? Nennen Sie mindestens vier Angaben.",
            "de",
            [["wohnungsgeber"], ["einzug"], ["anschrift", "adresse"], ["name"]],
        ),
        EvalQuestion(
            "Q4-EN",
            "EN",
            "Is registering the address free of charge?",
            "en",
            [["free", "no fee", "keine"], ["registration", "anmeldung"]],
        ),
        EvalQuestion(
            "Q5-DE",
            "DE",
            "Welche Unterlagen braucht man für die Anmeldung im Bürgeramt? Nennen Sie mindestens zwei.",
            "de",
            [["personalausweis", "reisepass", "pass"], ["einzugsbestätigung", "wohnungsgeberbestätigung"]],
        ),
        EvalQuestion(
            "Q6-EN",
            "EN",
            "What is the Einzugsbestätigung and who issues it?",
            "en",
            [["landlord", "wohnungsgeber"], ["confirmation", "bestätigung"]],
        ),
        EvalQuestion(
            "Q7-DE",
            "DE",
            "Unter welchen Bedingungen kann man sich online anmelden?",
            "de",
            [["bundid"], ["ausweisapp", "online-ausweisfunktion", "eid"], ["18", "volljährig"]],
        ),
        EvalQuestion(
            "Q10-EN",
            "EN",
            "In which cases do you not have to register? Name two exceptions with time limits.",
            "en",
            [["exception", "do not have to", "not required", "ausnahme"], ["month", "months", "tage", "days"]],
        ),
        EvalQuestion(
            "Q12-DE",
            "DE",
            "Was ist eine Hauptwohnungserklärung und wann braucht man das Beiblatt zur Anmeldung?",
            "de",
            [["hauptwohnungserklärung"], ["beiblatt", "anmeldung"]],
        ),
    ]


def _language_hint_score(language_label: str, answer: str) -> int:
    """Estimate language adherence score.

    Args:
        language_label: Target language label (EN/DE/FA).
        answer: Model answer.

    Returns:
        Integer score from 0 to 2.
    """
    lowered = answer.lower()
    if language_label == "FA":
        has_fa_script = any("\u0600" <= ch <= "\u06FF" for ch in answer)
        return 2 if has_fa_script else 0

    if language_label == "DE":
        german_markers = [" der ", " die ", " und ", " ist ", " anmeldung", "bürgeramt", "wohnungs"]
        marker_hits = sum(1 for marker in german_markers if marker in f" {lowered} ")
        return 2 if marker_hits >= 2 else 1 if marker_hits == 1 else 0

    english_markers = [" the ", " and ", " is ", "registration", "you", "address"]
    marker_hits = sum(1 for marker in english_markers if marker in f" {lowered} ")
    return 2 if marker_hits >= 2 else 1 if marker_hits == 1 else 0


def _correctness_score(expected_groups: List[List[str]], answer: str) -> int:
    """Score correctness by counting satisfied keyword groups.

    Args:
        expected_groups: Keyword OR-groups.
        answer: Model answer.

    Returns:
        Integer score from 0 to 4.
    """
    lowered = answer.lower()
    matched_groups = 0
    for group in expected_groups:
        if any(keyword.lower() in lowered for keyword in group):
            matched_groups += 1

    if matched_groups == 0:
        return 0
    if matched_groups == 1:
        return 1
    if matched_groups == 2:
        return 2
    if matched_groups == 3:
        return 3
    return 4


def _citation_score(citations: List[Dict[str, Any]]) -> int:
    """Score citation quality with simple deterministic checks.

    Args:
        citations: Citation list from backend response.

    Returns:
        Integer score from 0 to 3.
    """
    if not citations:
        return 0

    score = 1
    first = citations[0] if isinstance(citations[0], dict) else {}
    source_uri = str(first.get("source_uri", "")).lower()
    snippet = str(first.get("snippet", "")).strip()

    if "service.berlin.de" in source_uri or "wohnsitz" in source_uri or ".pdf" in source_uri:
        score += 1
    if len(snippet) >= 40:
        score += 1

    return min(score, 3)


def _run_eval(questions: List[EvalQuestion]) -> Dict[str, Any]:
    """Execute evaluation queries and compute per-item and aggregate metrics.

    Args:
        questions: Evaluation question list.

    Returns:
        Evaluation result payload.
    """
    rows: List[Dict[str, Any]] = []

    for item in questions:
        response = _post(
            "/api/query",
            {
                "question": item.question_text,
                "top_k": 5,
                "chat_history": [],
                "response_language": item.response_language,
            },
            timeout=180,
        )

        answer = str(response.get("answer", "")).strip()
        citations = response.get("citations", []) if isinstance(response.get("citations"), list) else []

        correctness = _correctness_score(item.expected_groups, answer)
        citations_quality = _citation_score(citations)
        language_quality = _language_hint_score(item.language_label, answer)
        completeness = 1 if len(answer) >= 40 else 0

        total = correctness + citations_quality + language_quality + completeness
        rows.append(
            {
                "id": item.question_id,
                "language": item.language_label,
                "question": item.question_text,
                "answer": answer,
                "citations": citations,
                "scores": {
                    "correctness_0_4": correctness,
                    "citations_0_3": citations_quality,
                    "language_0_2": language_quality,
                    "completeness_0_1": completeness,
                    "total_0_10": total,
                },
            }
        )

    totals = [row["scores"]["total_0_10"] for row in rows]
    avg_total = round(sum(totals) / len(totals), 3) if totals else 0.0
    perfect_count = sum(1 for score in totals if score >= 9)

    return {
        "question_count": len(rows),
        "average_total_score_0_10": avg_total,
        "perfect_or_near_perfect_count": perfect_count,
        "rows": rows,
    }


def _build_markdown(ingest_summary: Dict[str, Any], evaluation: Dict[str, Any], pdf_path: Path) -> str:
    """Build human-readable markdown report.

    Args:
        ingest_summary: Source ingestion summary.
        evaluation: Evaluation result payload.
        pdf_path: Path to the ingested PDF.

    Returns:
        Markdown report text.
    """
    lines: List[str] = []
    lines.append("# Wohnsitz Anmeldung Evaluation Report")
    lines.append("")
    lines.append("## Scope")
    lines.append("- Source 1: service.berlin.de official page (Wohnung anmelden)")
    lines.append(f"- Source 2: {pdf_path.name}")
    lines.append("- Mode: `/api/query` multilingual evaluation")
    lines.append("")
    lines.append("## Ingestion Summary")
    lines.append(f"- Clear success: {ingest_summary.get('clear', {}).get('success')}")
    lines.append(f"- URL ingest success: {ingest_summary.get('web_ingest', {}).get('success')}")
    lines.append(f"- PDF ingest success: {ingest_summary.get('pdf_ingest', {}).get('success')}")
    lines.append("")
    lines.append("## Rubric")
    lines.append("- Correctness: 0-4 (expected key facts/terms)")
    lines.append("- Citation quality: 0-3 (presence, source relevance, snippet quality)")
    lines.append("- Language adherence: 0-2")
    lines.append("- Completeness: 0-1")
    lines.append("- Total: 0-10")
    lines.append("")
    lines.append("## Aggregate Result")
    lines.append(f"- Question count: {evaluation['question_count']}")
    lines.append(f"- Average total score: {evaluation['average_total_score_0_10']} / 10")
    lines.append(f"- Near-perfect answers (>=9/10): {evaluation['perfect_or_near_perfect_count']}")
    lines.append("")
    lines.append("## Detailed Results")
    lines.append("| ID | Lang | Total | Correct | Cite | LangScore | Complete |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in evaluation["rows"]:
        scores = row["scores"]
        lines.append(
            f"| {row['id']} | {row['language']} | {scores['total_0_10']} | "
            f"{scores['correctness_0_4']} | {scores['citations_0_3']} | "
            f"{scores['language_0_2']} | {scores['completeness_0_1']} |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("- This is a deterministic heuristic evaluation for project presentation.")
    lines.append("- Final academic review can include manual verification for borderline cases.")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    """Execute full eval pipeline and persist report artifacts."""
    args = _parse_args()

    # Allow overriding base URLs without threading it through every helper.
    global BASE_URL
    BASE_URL = str(args.base_url).rstrip("/")

    pdf_path = Path(args.pdf_path)
    report_md = Path(args.report_md)
    report_json = Path(args.report_json)

    ingest_summary = _ingest_sources(pdf_path=pdf_path, web_source=str(args.web_source))
    evaluation = _run_eval(_build_questions())

    report_json.write_text(
        json.dumps({"ingest": ingest_summary, "evaluation": evaluation}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_md.write_text(_build_markdown(ingest_summary, evaluation, pdf_path=pdf_path), encoding="utf-8")

    print(json.dumps({
        "report_markdown": str(report_md),
        "report_json": str(report_json),
        "average_score": evaluation["average_total_score_0_10"],
        "question_count": evaluation["question_count"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
