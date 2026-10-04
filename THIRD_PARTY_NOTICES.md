# Third-party notices

## Privasis website

Source: https://github.com/privasis/privasis.github.io
Reference commit: d1226b766025060418654d81da3ce3913ef649db

Copyright (c) 2026 Hyunwoo Kim. The upstream MIT license is preserved in LICENSE. `dist/static/css/index.css` is copied from this source. The page structure, resource button styling, sidebar rules, and footer are adapted from its HTML. The benchmark text, figures, and tables are replaced; scripts are rewritten for the features used here. No upstream analytics are included.

## Nerfies template

Source: https://github.com/nerfies/nerfies.github.io

Privasis credits Nerfies and retains its Creative Commons Attribution-ShareAlike 4.0 notice. We preserve both attributions and release our adapted website template under the same CC BY-SA 4.0 terms: https://creativecommons.org/licenses/by-sa/4.0/ (legal text: https://creativecommons.org/licenses/by-sa/4.0/legalcode). The MIT permissions applying to upstream files remain in place. This template notice does not relicense the paper, its figures, or third-party datasets.

## Bulma 0.9.1

Source: https://github.com/jgthms/bulma/tree/0.9.1
`dist/static/css/bulma.min.css` is distributed under the MIT license. Its embedded license header is preserved. See LICENSES/Bulma-MIT.txt for the full upstream notice.

## Research content

AgentPrivArena: Evaluating and Auditing Real-world AI Agent Privacy, Shouju Wang and Haopeng Zhang. The manuscript and its figures are included at the author's request. Original figure artwork is unchanged; each is cropped from the supplied PDF. Provenance, original captions and source hash are recorded in `dist/assets/figures/manifest.json`. Template licenses do not relicense the manuscript or its figures.

Numerical results are transcribed from Table 4 on PDF page 6, with the full values in `dist/static/data/results.json`. Author identities are verified against the exact-title listing at https://hpzhang94.github.io/research/. No conference acceptance is asserted.

## OpenHands

This repository includes and modifies the
[OpenHands Software Agent SDK](https://github.com/OpenHands/software-agent-sdk).
The OpenHands MIT license and its copyright notice are preserved in
[LICENSE](LICENSE), including:

> Copyright (c) 2026 OpenHands contributors

Retain that license when redistributing the included OpenHands code. Installed
dependencies and container images retain their own licenses; the root license
does not replace their terms.

## PrivacyLens evaluator code and prompts

[agentprivarena/base/evaluator.py](agentprivarena/base/evaluator.py) includes
evaluation prompts and logic adapted from the
[PrivacyLens project](https://github.com/SALT-NLP/PrivacyLens). Its comments
identify prompts preserved verbatim and prompts adapted for live-service
evaluation. The following notice reproduces the upstream
[MIT license](https://github.com/SALT-NLP/PrivacyLens/blob/main/LICENSE):

```text
MIT License

Copyright (c) 2024 Stanford Social and Language Technologies (SALT) lab
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The associated publication is *PrivacyLens: Evaluating Privacy Norm Awareness
of Language Models in Action*, Yijia Shao, Tianshi Li, Weiyan Shi, Yanchen Liu,
and Diyi Yang (2024), [arXiv:2409.00138](https://arxiv.org/abs/2409.00138).

## External benchmark data and images

The public export does not bundle the PrivacyLens or MPCI-Bench datasets,
generated task payloads, or VISPR images. A private development checkout may
contain these artifacts. Obtain external data from its original distributors
and review the terms for the version you use.

- The official [PrivacyLens dataset card](https://huggingface.co/datasets/SALT-NLP/PrivacyLens)
  declares [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). This is
  separate from the PrivacyLens code license above. Attribute the dataset and
  identify any filtering, conversion, or other changes when redistributing
  derived data. The dataset card also asks users not to use it directly for
  training.
- The [MPCI-Bench repository](https://github.com/hpzhang94/MPCI-Bench) identifies
  its dataset as CC BY 4.0 and its code as MIT. Its images are an external
  dependency with separate terms.
- The [VISPR release](https://tribhuvanesh.github.io/vpa/) supplies source
  images used by MPCI-Bench. The MPCI-Bench documentation describes their
  original Flickr Creative Commons licensing and does not redistribute them.
  Check the applicable image-specific terms and attribution requirements;
  neither this repository's MIT license nor a benchmark's data license
  relicenses those images.

These notices document upstream attribution and published terms. They do not
assert that a particular derived dataset, annotation set, or image collection
has been reviewed or approved for redistribution.

## Project citation

This project's paper is *AgentPrivArena: Evaluating and Auditing Real-world AI
Agent Privacy*, by Shouju Wang and Haopeng Zhang, as listed on the
[author's publication page](https://hpzhang94.github.io/research/).
See [CITATION.cff](CITATION.cff). An arXiv identifier is not inferred from the
anonymous manuscript; the upstream PrivacyLens citation is a separate work.
