# AgentPrivArena project page

Project website for **AgentPrivArena: Evaluating and Auditing Real-world AI Agent Privacy**.

- Website: https://voidreaming.github.io/agentprivarena/
- Related project: https://voidreaming.github.io/mpci-bench/
- OpenReview: https://openreview.net/forum?id=zxllNfqsYS

The site directly shares the Privasis/Nerfies-based template, centered header, type sizes, figures, result tables, and More Research menu used by MPCI-Bench. GitHub Pages publishes `dist/` on pushes to `main`.

## Preview

```sh
python3 -m http.server 8766 --bind 127.0.0.1 --directory dist
```

## Content and provenance

The author supplied the manuscript in `dist/assets/agentprivarena.pdf` (15 pages). Its cover is an anonymous ACL submission; this site makes no conference-acceptance claim. The exact title and named authors are also listed on https://hpzhang94.github.io/research/. Current author affiliations come from their personal homepages. The citation year is the manuscript file year (PDF creation metadata: August 4, 2026), not an acceptance date.

Original figures are cropped from the supplied PDF with PyMuPDF and retain their original artwork. `dist/assets/figures/manifest.json` records page numbers, crop bounds, captions and the source PDF hash. `dist/static/data/results.json` contains all Table 4 results; no rounded values have been recomputed from displayed averages. Outcome leakage uses task counts; exposure and extraction use item counts; disposition uses flow counts.

Code and dataset buttons are intentionally disabled placeholders requested by the author. Replace those buttons with links in `dist/index.html` when public URLs are available. The footer's website-source link refers only to this website, not to the benchmark implementation.

## Editing

- `dist/index.html`: narrative, resource links, tables, citation and research menu.
- `dist/static/css/mpci.css`: shared visual style inherited from the MPCI-Bench page.
- `dist/static/css/arena.css`: AgentPrivArena-specific layouts.
- `dist/static/js/mpci.js`: sidebar, figure dialog, citation copying.
- `dist/static/js/research-nav.js`: keyboard and outside-click behavior for More Research.

All asset paths are relative and work under a GitHub Pages project URL. Update the More Research links on both sites if either project moves. See THIRD_PARTY_NOTICES.md and LICENSE for template credits; these do not relicense the paper.
