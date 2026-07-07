#!/usr/bin/env python3
"""Papers digest.

One Telegram message every Saturday morning (~9:07 IST via GitHub
Actions): the week's most relevant AI/data papers from arXiv, picked and
summarized for a data engineer building LLM systems.

Weekly on purpose — a daily arXiv feed is noise. Volume in is large
(~40 recent papers per category), volume out is small (6-8 picks).

One agent, one task, one bot.
"""

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
MAX_PER_CATEGORY = 40
ABSTRACT_CHARS = 600
LOOKBACK_DAYS = 7
INTERESTS = (
    "LLM systems and agents, RAG and vector search, model efficiency and "
    "inference, data engineering (pipelines, query engines, streaming), "
    "evaluation methods"
)


def fetch_recent(category, cutoff):
    """This week's submissions in one arXiv category (Atom API)."""
    r = requests.get(
        API,
        params={
            "search_query": f"cat:{category}",
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": MAX_PER_CATEGORY,
        },
        timeout=60,
    )
    r.raise_for_status()
    out = []
    for e in feedparser.parse(r.text).entries:
        stamp = e.get("published_parsed")
        if not stamp or datetime(*stamp[:6], tzinfo=timezone.utc) < cutoff:
            continue
        out.append(
            {
                "title": " ".join(e.get("title", "").split()),
                "abstract": " ".join(e.get("summary", "").split())[:ABSTRACT_CHARS],
                "link": e.get("link", ""),
            }
        )
    return out


def main():
    load_dotenv(BASE_DIR / ".env")
    cutoff = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)

    papers, failed, seen = [], [], set()
    for cat in CATEGORIES:
        try:
            for p in fetch_recent(cat, cutoff):
                if p["title"] not in seen:  # papers cross-list categories
                    seen.add(p["title"])
                    papers.append(p)
        except Exception as exc:  # one category failing must not kill the digest
            failed.append(f"{cat} ({type(exc).__name__})")

    header = (
        f"📄 Papers digest — week ending {datetime.now(IST):%d %b %Y}\n"
        f"({len(papers)} papers scanned)\n\n"
    )
    if not papers:
        body = "arXiv unreachable this week — nothing scanned ☕"
    else:
        blob = "\n\n".join(
            f"- {p['title']}\n  {p['abstract']}\n  {p['link']}" for p in papers
        )
        body = ask_llm(
            "Below are this week's arXiv submissions (title, abstract, "
            f"link). I am a data engineer; my interests: {INTERESTS}.\n\n"
            f"{blob}\n\n"
            "Pick the 6-8 papers most relevant to my interests, ranked. For "
            "each: the title on one line, then 2 terse sentences — what it "
            "shows and why it matters to someone building LLM/data systems "
            "— then the link on its own line. Blank line between papers. "
            "Plain text, no markdown. Skip pure theory unless the result "
            "is striking.",
            max_tokens=2000,
        )
    if failed:
        body += "\n\n⚠️ Could not check: " + ", ".join(failed)
    send_telegram(header + body)


if __name__ == "__main__":
    main()
