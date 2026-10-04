# AgentPrivArena project page

Project website for **AgentPrivArena: Evaluating and Auditing Real-world AI Agent Privacy**.

- Website: https://shouju-wang.github.io/agentprivarena/
- Website source: https://github.com/shouju-wang/shouju-wang.github.io/tree/master/agentprivarena
- Related project: https://shouju-wang.github.io/mpci-bench/
- OpenReview: https://openreview.net/forum?id=zxllNfqsYS

The site directly shares the Privasis/Nerfies-based template, centered header, type sizes, figures, result tables, and More Research menu used by MPCI-Bench. The active website is hosted by the personal-site repository on its `master` branch. This repository continues to host the research code and preserves the former website URL as a redirect.

## Preview the active site

```sh
git clone https://github.com/shouju-wang/shouju-wang.github.io.git personal-profile
python3 -m http.server 8767 --bind 127.0.0.1 --directory personal-profile
```

Open `http://localhost:8767/agentprivarena/`.

## Content and provenance

The author supplied the manuscript in `dist/assets/agentprivarena.pdf` (15 pages). Its cover is an anonymous ACL submission; this site makes no conference-acceptance claim. The exact title and named authors are also listed on https://hpzhang94.github.io/research/. Current author affiliations come from their personal homepages. The citation year is the manuscript file year (PDF creation metadata: August 4, 2026), not an acceptance date.

Original figures are cropped from the supplied PDF with PyMuPDF and retain their original artwork. `dist/assets/figures/manifest.json` records page numbers, crop bounds, captions and the source PDF hash. `dist/static/data/results.json` contains all Table 4 results; no rounded values have been recomputed from displayed averages. Outcome leakage uses task counts; exposure and extraction use item counts; disposition uses flow counts.

The Code button links to this repository, which now includes the SDK research
fork and benchmark implementation. The Dataset button remains a disabled
placeholder until a separate data release is available. The footer's website-source link points to the personal-site repository; the Code button continues to point to the research implementation here.

## Editing and publishing

Edit `agentprivarena/index.html`, `agentprivarena/static/`, and
`agentprivarena/assets/` in the personal-site repository. Push its `master`
branch to publish website changes. Keep the More Research links aligned with
`https://shouju-wang.github.io/mpci-bench/` and
`https://shouju-wang.github.io/agentprivarena/`.

This repository's `dist/index.html` redirects visitors to the personal site,
preserving query strings and section anchors with JavaScript and providing a
no-JavaScript refresh and visible link. Existing `dist/assets/` and
`dist/static/` files remain available for previously shared direct URLs and
README figures. Keep the Pages workflow enabled; pushes to `main` continue to
publish this redirect and those assets. The research code remains in this
repository.

See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) and [LICENSE](../LICENSE)
for source credits and licenses; template notices do not relicense the paper.
