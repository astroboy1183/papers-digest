# papers-digest

Weekly arXiv digest → Telegram, Saturday 6:00 IST via GitHub Actions.
The week's most relevant papers for a data & AI engineer — picked from
~2,500 submissions across seven categories, grouped into four interest
sections, each with two terse sentences and its arXiv link, plus a
📌 SPOTLIGHT deep-dive on the week's #1. One agent, one task, one bot.

```
📄 Papers digest — week ending 11 Jul 2026
(1,004 papers scanned across 7 categories, 61 read in full, 14 trending on HF)

🤖 AI & LLM
🔥 <title>
<what it shows; why it matters.>
https://arxiv.org/abs/…

👁 VISION
…

🗄 DATA & SYSTEMS
…

🔧 HARDWARE
…

📌 SPOTLIGHT — <the #1 pick>
<5-6 sentences: method, headline numbers, admitted limitations, why it
matters in practice.>
https://arxiv.org/abs/…
```

## How the code works

`papers_digest.py`, in pipeline order:

- **`CATEGORIES`** — cs.LG, cs.CL, cs.AI (core ML/LLM), **cs.CV**
  (vision), cs.DB, **cs.DC** (data & systems), **cs.AR** (hardware
  architecture — accelerators, chips). `fetch_recent()` pages each
  category newest-first until it crosses the 7-day cutoff (bounded by
  `MAX_PAGES`), then down-samples evenly across the week to 150/category
  so high-volume ML can't crowd out data/hardware.
- **`interests()`** — from the `PAPERS_INTERESTS` secret (LLM systems,
  computer vision, data engineering, ML infrastructure, hardware
  accelerators, evaluation…); change anytime with `gh secret set`.
- **`hf_upvotes()`** — Hugging Face's daily-papers API for the window:
  `{arxiv id: upvotes}`, merged deterministically. Papers at ≥20
  upvotes are 🔥-flagged for both models AND **force-included in the
  shortlist** — a skim cannot drop what the community is talking about.
  An unreachable API costs the signal, never the digest.
- **`shortlist()`** — stage 1 (haiku): skims title+snippet in chunks of
  150, keeps up to 12/chunk, recall over precision; an unparseable
  chunk is kept whole.
- **`rank()`** — stage 2 (sonnet): reads the survivors' full abstracts,
  picks 10-12 grouped under 🤖 AI & LLM / 👁 VISION / 🗄 DATA & SYSTEMS /
  🔧 HARDWARE (2-4 each, section skipped only when empty), and appends a
  `===STATE===` tail naming the picked arxiv ids, best first.
- **`validate_links()`** — every URL in the digest must be one we handed
  the model; anything else becomes `(link unavailable)` — a mangled
  arXiv id silently points at someone else's paper.
- **`spotlight()`** — the #1 pick, read for real: arXiv's HTML full text
  (up to 15k chars, abstract fallback) → 5-6 sentences on method,
  numbers, limitations, practical relevance. Best-effort enrichment.
- **Served memory** (`state/served.json`, committed back) — picked ids
  kept 90 days and excluded from future pools, so a revised
  resubmission is never served twice.
- **`agentlib.py`** (vendored) — `ask_llm()`, `send_telegram()`.

## Design notes

- Weekly on purpose — a daily arXiv feed is noise; Saturday morning is
  reading time.
- Two-stage because one model can't judge ~1,000 abstracts in one call;
  two tiers can, for pennies. `PAPERS_MODEL_FILTER` /
  `PAPERS_MODEL_RANK` override the tiers.
- Silent only when the week is genuinely thin AND nothing failed; fetch
  failures raise loudly instead.
- Tests run in CI on every push (`.github/workflows/tests.yml`).

## Ops

- Schedule: fleet-scheduler dispatches Sat 06:00 IST; backup cron
  `30 1 * * 6` UTC (Sat 07:00 IST) with the dedupe guard.
- Run now: `gh workflow run papers-digest.yml -R astroboy1183/papers-digest`
- Secrets (Actions): `ANTHROPIC_API_KEY`, `TELEGRAM_BOT_TOKEN`,
  `TELEGRAM_CHAT_ID`, `PAPERS_INTERESTS` (optional; defaults in code).
