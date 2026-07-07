# papers-digest

Weekly arXiv digest → Telegram, Saturdays ~9:07 AM IST via GitHub
Actions. One agent, one task, one bot.

The week's AI/data papers from four arXiv categories, filtered down to
the 6-8 most relevant to a data engineer building LLM systems — what
each shows, why it matters, and the link.

## How the code works

`papers_digest.py`, in pipeline order:

- **`CATEGORIES`** — `cs.LG`, `cs.CL`, `cs.AI`, `cs.DB`. **`INTERESTS`**
  — the relevance filter the model applies (LLM systems/agents, RAG and
  vector search, efficiency, data engineering, evals). Edit either to
  retune the digest.
- **`fetch_recent(category, cutoff)`** — one call per category to the
  arXiv Atom API (`export.arxiv.org/api/query`), newest first, capped at
  40; entries older than 7 days are dropped. Abstracts truncated to 600
  chars to keep the prompt sane. feedparser parses the Atom response
  like any feed.
- **`main()`** — dedupes across categories by title (papers cross-list),
  with per-category `try/except` feeding a "⚠️ Could not check" footer.
  One model call picks and ranks the 6-8 most relevant: title, 2 terse
  sentences (what it shows, why it matters to me), link. arXiv fully
  unreachable → a one-line message, no model call.
- **`agentlib.py`** (vendored) — `ask_llm()` one-shot model call;
  `send_telegram()` chunked sends.

## Design notes

- Weekly on purpose: daily arXiv is noise; ~160 candidates in, 6-8
  picks out, once a week, is readable.
- Like release-radar: no backup cron — a dropped Saturday is covered by
  next week's 7-day lookback. Failure pings this bot directly.

## Ops

- Schedule: `.github/workflows/papers-digest.yml` (`37 3 * * 6` UTC = Sat 09:07 IST)
- Run now: `gh workflow run papers-digest.yml -R astroboy1183/papers-digest`
- Secrets (Actions): `ANTHROPIC_API_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`
