# Integrated measured-results manuscript

**Rebalancing Uneven Visual Data: Feature Equalization, Sampling, and Synthetic Images**

Savannah Shannon, Kishor Datta Gupta, and Roy George.

This manuscript integrates the completed 80-tuning/720-final-fit primary study,
the separate 48-fit exploratory EQ follow-up, and the separate 192-fit generated-image
follow-up. It opens with an illustrative biodiversity cataloguing case; no deployment
in that setting is claimed. The original prospective protocol remains at
[`../main.tex`](../main.tex).

Open [the compiled PDF](REBAL_Integrated_Study_20261103.pdf), or compile `main.tex`
with pdfLaTeX (two passes). All three figure PDFs are included. The source was saved
and compiled in the existing Overleaf project. This is an IEEE-style research draft,
not an accepted or submitted CVPR paper.

[Per-seed data, complete comparison families, protocols, and reproduction script](../../results/20261102/README.md).

The central finding is conditional: EQ's direction depends on the sampler and
setting, while generated additions beat count-matched duplicates under a fixed
adapter-update budget with diminishing returns over the tested doses. The stronger
calibrated cross-entropy reference remains visible. The study does not establish a
universal limit of synthetic data, an optimal dose, or a causal mechanism.
