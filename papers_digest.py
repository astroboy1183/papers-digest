#!/usr/bin/env python3
"""Papers digest.

One Telegram message every Saturday morning (~9:07 IST via GitHub
Actions): the week's most relevant AI/data papers from arXiv, picked and
summarized for a data engineer building LLM systems.

Weekly on purpose — a daily arXiv feed is noise. Volume in is large
(a busy category like cs.LG sees ~750 submissions/week), volume out is
small (6-8 picks).

Two-stage review: a cheap model skims every candidate (title + snippet,
chunked) and shortlists the ones worth reading in full; a stronger model
then ranks the shortlist on complete abstracts. One model can't judge
~600 abstracts in one call — two tiers can, for pennies.

One agent, one task, one bot.
"""

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import feedparser
import requests
from dotenv import load_dotenv

from agentlib import ask_llm, send_telegram

BASE_DIR = Path(__file__).resolve().parent
IST = ZoneInfo("Asia/Kolkata")

API = "http://export.arxiv.org/api/query"
CATEGORIES = ["cs.LG", "cs.CL", "cs.AI", "cs.DB"]
PAGE_SIZE = 100  # arXiv results per API request
MAX_PAGES = 12  # hard cap on paging (<=1200 entries/category) — no runaway loops
MAX_PER_CATEGORY = 150  # candidates handed to the model per category (see below)
ABSTRACT_CHARS = 600
LOOKBACK_DAYS = 7
MIN_PAPERS = 3  # below this, stay silent rather than send filler
# Stage-1 skim: chunk size, keeps per chunk, and how much abstract the
# cheap model sees. Stage 2 reads the survivors' full abstracts.
FILTER_CHUNK = 150
FILTER_KEEP = 12
SNIPPET_CHARS = 200
INTERESTS = (
    "LLM systems and agents, RAG and vector search, model efficiency and "
    "inference, data engineering (pipelines, query engines, streaming), "
    "evaluation methods"
)


def fetch_recent(category, cutoff):
    """This week's submissions in one arXiv category, paginated (Atom API).

    arXiv sorts newest-first and caps each response, so a busy category needs
    several pages to reach the far end of the 7-day window — a single 40-result
    call only ever saw the newest day and silently dropped days 2-7. Page with
    `start=` until an entry predates the cutoff (newest-first, so everything
    after it is older too) or the feed runs dry, bounded by MAX_PAGES.

    A full week of a busy category is ~750 papers; four categories together run
    to ~1500, whose abstracts would blow past the model's context window. So the
    per-category pool is down-sampled to MAX_PER_CATEGORY *evenly across the
    week* — keeping a spread from every day, not just the newest — which both
    fits the single model call and keeps high-volume ML categories from
    crowding out data-engineering ones. (A proper two-stage re-rank is a
    separate, supervised change.)
    """
    out = []
    for page in range(MAX_PAGES):
        if page:
            time.sleep(3)  # arXiv asks ~3s between requests; also dodges throttling
        r = requests.get(
            API,
            params={
                "search_query": f"cat:{category}",
                "sortBy": "submittedDate",
                "sortOrder": "descending",
                "start": page * PAGE_SIZE,
                "max_results": PAGE_SIZE,
            },
            timeout=60,
        )
        r.raise_for_status()
        entries = feedparser.parse(r.text).entries
        if not entries:
            break  # ran past the end of the category feed
        reached_cutoff = False
        for e in entries:
            stamp = e.get("published_parsed")
            if not stamp:
                continue
            if datetime(*stamp[:6], tzinfo=timezone.utc) < cutoff:
                reached_cutoff = True
                break  # newest-first: everything after this is older too
            prim = e.get("arxiv_primary_category") or {}
            out.append(
                {
                    "title": " ".join(e.get("title", "").split()),
                    "abstract": " ".join(e.get("summary", "").split())[:ABSTRACT_CHARS],
                    "link": e.get("link", ""),
                    "category": prim.get("term", "") if hasattr(prim, "get") else "",
                }
            )
        if reached_cutoff:
            break

    if len(out) > MAX_PER_CATEGORY:
        step = len(out) / MAX_PER_CATEGORY
        out = [out[int(i * step)] for i in range(MAX_PER_CATEGORY)]
    return out


def shortlist(papers, model):
    """Stage 1: a cheap model skims title + snippet and keeps candidates.

    Chunked so each call stays small and reliably parseable. Prefers
    recall over precision — stage 2 does the real judging. A chunk whose
    reply can't be parsed is kept whole: handing stage 2 too much beats
    silently dropping a chunk."""
    kept = []
    for i in range(0, len(papers), FILTER_CHUNK):
        chunk = papers[i : i + FILTER_CHUNK]
        listing = "\n".join(
            f"{j}. [{p['category']}] {p['title']} — {p['abstract'][:SNIPPET_CHARS]}"
            for j, p in enumerate(chunk)
        )
        reply = ask_llm(
            f"I am a data engineer; my interests: {INTERESTS}.\n\n"
            "Below are arXiv papers (index. [category] title — abstract "
            f"snippet). Return a JSON array of the indices of up to "
            f"{FILTER_KEEP} papers a reviewer should read in full to judge "
            "relevance to my interests. Prefer recall over precision — when "
            "unsure, include. Output ONLY the JSON array, nothing else.\n\n"
            + listing,
            max_tokens=300,
            model=model,
        )
        try:
            start, end = reply.find("["), reply.rfind("]")
            idx = json.loads(reply[start : end + 1])
            kept += [
                chunk[j] for j in idx if isinstance(j, int) and 0 <= j < len(chunk)
            ]
        except (ValueError, TypeError):
            kept += chunk
    return kept


def main():
    load_dotenv(BASE_DIR / ".env")
    cutoff = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)
    # env read after load_dotenv so .env values work too
    filter_model = os.environ.get("PAPERS_MODEL_FILTER") or "claude-haiku-4-5"
    rank_model = os.environ.get("PAPERS_MODEL_RANK") or "claude-sonnet-5"

    papers, failed, seen = [], [], set()
    for cat in CATEGORIES:
        try:
            for p in fetch_recent(cat, cutoff):
                if p["title"] not in seen:  # papers cross-list categories
                    seen.add(p["title"])
                    papers.append(p)
        except Exception as exc:  # one category failing must not kill the digest
            failed.append(f"{cat} ({type(exc).__name__})")

    if len(papers) < MIN_PAPERS:
        # Nothing worth a digest. A genuinely quiet (but successful) week stays
        # silent per the fleet convention — no filler "nothing scanned" message.
        # If the thinness is because fetches actually failed, surface that
        # loudly via the workflow's failure-alert path instead.
        if failed:
            raise RuntimeError("arXiv fetch failed for: " + ", ".join(failed))
        return

    # Stage 1: skim everything cheaply; stage 2 judges the survivors on
    # their full abstracts. A filter that kept nothing is a broken filter,
    # so fall back to the full pool rather than go silent.
    finalists = shortlist(papers, filter_model) if len(papers) > FILTER_KEEP else papers
    if not finalists:
        finalists = papers

    blob = "\n\n".join(
        f"- [{p['category']}] {p['title']}\n  {p['abstract']}\n  {p['link']}"
        for p in finalists
    )
    body = ask_llm(
        "Below are this week's arXiv submissions that already passed a "
        "first relevance skim (primary category in brackets, then title, "
        "full abstract, link). I am a data engineer; my interests: "
        f"{INTERESTS}.\n\n"
        f"{blob}\n\n"
        "Pick the 6-8 papers most relevant to my interests, ranked. Balance "
        "the selection across topics — don't let the high-volume ML categories "
        "(cs.LG, cs.CL, cs.AI) crowd out data-engineering work (cs.DB). For "
        "each: the title on one line, then 2 terse sentences — what it shows "
        "and why it matters to someone building LLM/data systems — then the "
        "link on its own line. Blank line between papers. Plain text, no "
        "markdown. Skip pure theory unless the result is striking.",
        max_tokens=2000,
        model=rank_model,
    )

    header = (
        f"📄 Papers digest — week ending {datetime.now(IST):%d %b %Y}\n"
        f"({len(papers)} papers scanned, {len(finalists)} read in full)\n\n"
    )
    if failed:
        body += "\n\n⚠️ Could not check: " + ", ".join(failed)
    send_telegram(header + body)


if __name__ == "__main__":
    main()
