#!/usr/bin/env python3
"""Papers digest.

One Telegram message every Saturday morning (6:00 IST via GitHub
Actions): the week's most relevant papers from arXiv across MY interests
— AI/LLM systems, computer vision, data & systems, and hardware — picked,
sectioned and summarized, with a 📌 SPOTLIGHT deep-dive on the week's #1.

Weekly on purpose — a daily arXiv feed is noise. Volume in is large
(~2,500 submissions/week across seven categories), volume out is small
(10-12 picks in four sections).

Two-stage review: a cheap model skims every candidate (title + snippet,
chunked) and shortlists the ones worth reading in full; a stronger model
then ranks the shortlist on complete abstracts. One model can't judge
~1,000 abstracts in one call — two tiers can, for pennies.

Community signal: Hugging Face's daily-papers upvotes are merged in
deterministically — trending papers are 🔥-flagged for the ranker and
force-included in the shortlist, so the week's talked-about papers can
never be skimmed away.

Memory (state/served.json, committed back by the workflow): picked
papers are remembered for 90 days, so a revised resubmission is never
served twice.

One agent, one task, one bot.
"""

import json
import os
import re
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
HF_API = "https://huggingface.co/api/daily_papers"
# cs.LG/CL/AI: core ML & LLM · cs.CV: vision · cs.DB/DC: data & systems ·
# cs.AR: hardware architecture (accelerators, chips)
CATEGORIES = ["cs.LG", "cs.CL", "cs.AI", "cs.CV", "cs.DB", "cs.DC", "cs.AR"]
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
HF_HOT = 20          # upvotes at/above this = 🔥, force-included in stage 2
SPOTLIGHT_CHARS = 15000  # of the #1 paper's full HTML text for the deep-dive

STATE_DIR = BASE_DIR / "state"
SERVED_FILE = STATE_DIR / "served.json"
SERVED_DAYS = 90  # long enough that a v2 resubmission never repeats
STATE_MARKER = "===STATE==="

ARXIV_ID_RE = re.compile(r"/abs/([0-9]{4}\.[0-9]{4,5})")
URL_RE = re.compile(r"https?://\S+")


def validate_links(text, known_links):
    """Neutralize invented URLs: every link in the digest must be one we
    actually handed the model — a mangled arXiv id silently points at
    someone else's paper, which is worse than no link."""

    def check(match):
        url = match.group(0)
        trail = ""
        while url and url[-1] in ").,;'\"":
            trail = url[-1] + trail
            url = url[:-1]
        if url in known_links:
            return match.group(0)
        return "(link unavailable)" + trail

    return URL_RE.sub(check, text)

# Message sections, in display order. The ranker assigns each pick.
SECTIONS = "🤖 AI & LLM\n👁 VISION\n🗄 DATA & SYSTEMS\n🔧 HARDWARE"


def interests():
    """My paper interests, from the PAPERS_INTERESTS secret."""
    return os.environ.get("PAPERS_INTERESTS") or (
        "LLM systems and agents, RAG and vector search, computer vision, "
        "model efficiency and inference, data engineering (pipelines, "
        "query engines, streaming), ML infrastructure, hardware "
        "accelerators and chips, evaluation methods"
    )


def arxiv_id(link):
    """The bare arXiv id from an abs link ('2507.01234'), '' if none."""
    m = ARXIV_ID_RE.search(link or "")
    return m.group(1) if m else ""


def load_served():
    """{arxiv id: 'YYYY-MM-DD'} of papers already served, pruned."""
    try:
        served = json.loads(SERVED_FILE.read_text())
    except (OSError, ValueError):
        return {}
    cutoff = (datetime.now(timezone.utc) - timedelta(days=SERVED_DAYS)).strftime(
        "%Y-%m-%d"
    )
    return {k: v for k, v in served.items() if isinstance(v, str) and v >= cutoff}


def save_served(served):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    SERVED_FILE.write_text(json.dumps(served, indent=0, sort_keys=True) + "\n")


def split_state(reply):
    """(message text, picked arxiv ids) from the ranker's reply.

    A malformed tail costs the served memory, never the digest."""
    if STATE_MARKER not in reply:
        return reply.strip(), []
    text, _, tail = reply.partition(STATE_MARKER)
    start, end = tail.find("{"), tail.rfind("}")
    ids = []
    if start != -1 and end > start:
        try:
            ids = json.loads(tail[start : end + 1]).get("picked", [])
        except (ValueError, AttributeError):
            ids = []
    return text.strip(), [i for i in ids if isinstance(i, str)]


def hf_upvotes(days=LOOKBACK_DAYS):
    """{arxiv id: upvotes} from Hugging Face's daily-papers lists for the
    window — the community's 'what actually matters this week' signal.
    {} on any failure: an enrichment, never a dependency."""
    votes = {}
    for d in range(days):
        day = (datetime.now(timezone.utc) - timedelta(days=d)).strftime("%Y-%m-%d")
        try:
            r = requests.get(HF_API, params={"date": day}, timeout=20)
            r.raise_for_status()
            for item in r.json():
                paper = item.get("paper") or {}
                pid, up = paper.get("id"), paper.get("upvotes")
                if pid and isinstance(up, int):
                    votes[pid] = max(votes.get(pid, 0), up)
        except Exception:
            continue  # one missing day must not cost the signal
    return votes


def fetch_recent(category, cutoff):
    """This week's submissions in one arXiv category, paginated (Atom API).

    arXiv sorts newest-first and caps each response, so a busy category needs
    several pages to reach the far end of the 7-day window. Page with `start=`
    until an entry predates the cutoff or the feed runs dry, bounded by
    MAX_PAGES.

    A full week of a busy category is ~750 papers; seven categories together
    run past 2,500, so the per-category pool is down-sampled to
    MAX_PER_CATEGORY *evenly across the week* — keeping a spread from every
    day, and keeping high-volume ML categories from crowding out the
    data/hardware ones.
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


def hot_flag(paper):
    up = paper.get("upvotes", 0)
    return f" 🔥HF:{up}" if up >= HF_HOT else ""


def shortlist(papers, model):
    """Stage 1: a cheap model skims title + snippet and keeps candidates.

    Chunked so each call stays small and reliably parseable. Prefers
    recall over precision — stage 2 does the real judging. A chunk whose
    reply can't be parsed is kept whole: handing stage 2 too much beats
    silently dropping a chunk. 🔥 papers are force-included afterwards
    regardless of what the skim decided."""
    kept = []
    for i in range(0, len(papers), FILTER_CHUNK):
        chunk = papers[i : i + FILTER_CHUNK]
        listing = "\n".join(
            f"{j}. [{p['category']}]{hot_flag(p)} {p['title']} — "
            f"{p['abstract'][:SNIPPET_CHARS]}"
            for j, p in enumerate(chunk)
        )
        reply = ask_llm(
            f"I am a data & AI engineer; my interests: {interests()}.\n\n"
            "Below are arXiv papers (index. [category] title — abstract "
            f"snippet; 🔥HF:N marks Hugging Face community upvotes). Return "
            f"a JSON array of the indices of up to {FILTER_KEEP} papers a "
            "reviewer should read in full to judge relevance to my "
            "interests. Prefer recall over precision — when unsure, "
            "include. Output ONLY the JSON array, nothing else.\n\n"
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
    # The community's picks always reach stage 2 — a skim must not be
    # able to drop a trending paper.
    kept_links = {p["link"] for p in kept}
    kept += [
        p for p in papers
        if p.get("upvotes", 0) >= HF_HOT and p["link"] not in kept_links
    ]
    return kept


def rank(finalists, model):
    """Stage 2: the strong model reads full abstracts and writes the
    sectioned digest, with a state tail naming the picked arxiv ids."""
    blob = "\n\n".join(
        f"- [{p['category']}]{hot_flag(p)} {p['title']}\n  {p['abstract']}\n"
        f"  {p['link']}"
        for p in finalists
    )
    return ask_llm(
        "Below are this week's arXiv submissions that already passed a "
        "first relevance skim (primary category in brackets, 🔥HF:N marks "
        "Hugging Face community upvotes, then title, full abstract, "
        f"link). I am a data & AI engineer; my interests: {interests()}.\n\n"
        f"{blob}\n\n"
        "Pick the 10-12 papers most relevant to my interests and group "
        "them under EXACTLY these section headers, in this order, "
        "skipping a section only when nothing fits:\n\n"
        f"{SECTIONS}\n\n"
        "Rules:\n"
        "- 2-4 papers per section, ranked within it. The high-volume ML "
        "categories must not crowd out vision, data or hardware picks.\n"
        "- 🔥 papers deserve extra weight — the community rarely upvotes "
        "junk — but relevance to MY interests wins over popularity.\n"
        "- For each paper: the title on one line (prefix 🔥 if HF-hot), "
        "then 2 terse sentences — what it shows and why it matters to "
        "someone building AI/data systems — then the link on its own "
        "line. Blank line between papers.\n"
        "- Plain text, no markdown. Skip pure theory unless the result "
        "is striking.\n\n"
        f"Then output the line {STATE_MARKER} and ONE JSON object: "
        '{"picked": [the arxiv ids of every paper you included, e.g. '
        '"2507.01234", BEST PAPER FIRST]}. No text after the JSON.',
        max_tokens=2500,
        model=model,
    )


def spotlight(paper, model):
    """📌 SPOTLIGHT: a real read of the week's #1 pick — method, numbers,
    limitations — from the paper's full HTML text where arXiv has it,
    falling back to the abstract. '' on any failure: an enrichment."""
    pid = arxiv_id(paper["link"])
    text = ""
    try:
        r = requests.get(
            f"https://arxiv.org/html/{pid}",
            timeout=30,
            headers={"User-Agent": "papers-digest/1.0"},
        )
        if r.ok:
            body = re.sub(r"(?is)<(script|style|head|nav)[^>]*>.*?</\1>", " ", r.text)
            text = " ".join(re.sub(r"<[^>]+>", " ", body).split())[:SPOTLIGHT_CHARS]
    except Exception:
        text = ""
    reply = ask_llm(
        "Write a SPOTLIGHT summary of this paper for me (a data & AI "
        "engineer): 5-6 sentences covering the method, the headline "
        "numbers, the limitations the text admits, and why it matters "
        "in practice. Be concrete; never invent numbers.\n\n"
        f"TITLE: {paper['title']}\n\n"
        f"TEXT: {text or paper['abstract']}",
        max_tokens=500,
        model=model,
    ).strip()
    if not reply:
        return ""
    return f"📌 SPOTLIGHT — {paper['title']}\n{reply}\n{paper['link']}"


def main():
    load_dotenv(BASE_DIR / ".env")
    cutoff = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)
    # env read after load_dotenv so .env values work too
    filter_model = os.environ.get("PAPERS_MODEL_FILTER") or "claude-haiku-4-5"
    rank_model = os.environ.get("PAPERS_MODEL_RANK") or "claude-sonnet-5"

    served = load_served()
    papers, failed, seen = [], [], set()
    for cat in CATEGORIES:
        try:
            for p in fetch_recent(cat, cutoff):
                pid = arxiv_id(p["link"])
                if p["title"] in seen or (pid and pid in served):
                    continue  # cross-listed, or already served a past week
                seen.add(p["title"])
                papers.append(p)
        except Exception as exc:  # one category failing must not kill the digest
            failed.append(f"{cat} ({type(exc).__name__})")

    if len(papers) < MIN_PAPERS:
        # Nothing worth a digest. A genuinely quiet (but successful) week stays
        # silent per the fleet convention. If the thinness is because fetches
        # actually failed, surface that loudly via the failure-alert path.
        if failed:
            raise RuntimeError("arXiv fetch failed for: " + ", ".join(failed))
        return

    # The community signal, merged deterministically.
    votes = hf_upvotes()
    for p in papers:
        p["upvotes"] = votes.get(arxiv_id(p["link"]), 0)
    hot = sum(1 for p in papers if p["upvotes"] >= HF_HOT)

    # Stage 1: skim everything cheaply; stage 2 judges the survivors on
    # their full abstracts. A filter that kept nothing is a broken filter,
    # so fall back to the full pool rather than go silent.
    finalists = shortlist(papers, filter_model) if len(papers) > FILTER_KEEP else papers
    if not finalists:
        finalists = papers

    body, picked = split_state(rank(finalists, rank_model))
    body = validate_links(body, {p["link"] for p in finalists if p["link"]})

    # 📌 the week's #1, read in full — best-effort enrichment.
    if picked:
        by_id = {arxiv_id(p["link"]): p for p in finalists}
        top = by_id.get(picked[0])
        if top:
            try:
                spot = spotlight(top, rank_model)
            except Exception:
                spot = ""
            if spot:
                body += "\n\n" + spot

    header = (
        f"📄 Papers digest — week ending {datetime.now(IST):%d %b %Y}\n"
        f"({len(papers)} papers scanned across {len(CATEGORIES)} categories, "
        f"{len(finalists)} read in full, {hot} trending on HF)\n\n"
    )
    if failed:
        body += "\n\n⚠️ Could not check: " + ", ".join(failed)
    send_telegram(header + body)

    # Remember the picks — after the send, so a state failure never costs
    # the digest. A revised resubmission is never served twice.
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    for pid in picked:
        served[pid] = today
    try:
        save_served(served)
    except OSError:
        pass


if __name__ == "__main__":
    main()
