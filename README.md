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
- **`fetch_recent(category, cutoff)`** — **paginated** calls per category
  to the arXiv Atom API (`export.arxiv.org/api/query`), newest first, in
  pages of 100 with `start=`. arXiv sorts newest-first and caps each
  response, so a busy category (cs.LG sees ~750 submissions/week) needs
  several pages to reach the far end of the window — the old single
  40-result call only ever saw the newest day and silently dropped days
  2-7. Paging stops when an entry predates the 7-day cutoff (everything
  after is older too) or the feed runs dry, hard-bounded by `MAX_PAGES`
  (12 → ≤1200 entries/category), with a 3s pause between pages per arXiv's
  rate guidance. A full week across four categories is ~1500 papers —
  too many abstracts for the model's context window — so each category's
  pool is down-sampled to `MAX_PER_CATEGORY` (150) *evenly across the
  week*, keeping a spread from every day and stopping the high-volume ML
  categories from crowding out cs.DB. Each paper carries its arXiv primary
  category tag. Abstracts truncated to 600 chars. (A two-stage
  full-abstract re-rank is deferred as a separate, supervised change.)
- **`main()`** — dedupes across categories by title (papers cross-list),
  with per-category `try/except` feeding a "⚠️ Could not check" footer.
  One model call picks and ranks the 6-8 most relevant, told to balance
  across topics using the category tags: title, 2 terse sentences (what it
  shows, why it matters to me), link. A genuinely thin week (< 3 papers
  and no fetch failures) sends **nothing** — silence over filler, per the
  fleet convention. If the week is thin *because* fetches failed, it
  raises so the workflow's failure-alert step pings loudly.
- **`agentlib.py`** (vendored) — `ask_llm()` one-shot model call;
  `send_telegram()` chunked sends.

## Design notes

- Weekly on purpose: daily arXiv is noise; a few hundred candidates in
  (down-sampled from the full ~1500-paper week), 6-8 picks out, once a
  week, is readable.
- Like release-radar: no backup cron — a dropped Saturday is covered by
  next week's 7-day lookback. A failed run pings via the workflow's
  failure-alert step.

## Ops

- Schedule: `.github/workflows/papers-digest.yml` (`37 3 * * 6` UTC = Sat 09:07 IST)
- Run now: `gh workflow run papers-digest.yml -R astroboy1183/papers-digest`
- Secrets (Actions): `ANTHROPIC_API_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`
