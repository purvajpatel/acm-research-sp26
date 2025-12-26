# What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?
## https://proceedings.neurips.cc/paper_files/paper/2017/file/2650d6089a6d640c5e85b2b88265dc2b-Paper.pdf

## Research Motivation and Overview
Modern deep learning models usually provide predictions withouth indicating if the uncertainity arises from data noise or insufficient model knowledge (lack of data). This paper addresses this particualr limitation by splitting predictive uncertainity into aleatoric and epistemic., Aleatoric uncertainty capures data noise while epistemic uncertainty captures uncertainity from lack of data and model capacity. This paper provides a framework that combines heterosedastic likelihood modeling with Monte Carlo dropouts to estimate and decomes these types in deep nueral networks. 

## Advantages and  Limitations

**Advantages**
- Explicitly decomposes uncertainty sources rather than reporting a single aggregate confidence.
- Epistemic uncertainty highlights out-of-distribution inputs and data-sparse regions.
- Aleatoric uncertainty adapts to input-dependent noise without requiring labeled uncertainty.
- Requires minimal architectural changes and integrates with standard deep learning workflows.

**Limitations**
- Monte Carlo Dropout provides an approximation to Bayesian inference and may underestimate uncertainty.
- Aleatoric uncertainty assumes noise can be modeled with simple parametric distributions.
- Computational cost scales with the number of Monte Carlo samples.
- Quality of uncertainty estimates depends on dropout placement and training stability.

## Novelty & Key Takeaways
The central contribution is not merely estimating uncertainty, but distinguishing the source of uncertainty. By jointly modeling aleatoric and epistemic uncertainty, the method enables interpretable risk assessment and principled decision-making under uncertainty. This distinction is particularly valuable in finance and other high-stakes domains, where separating market noise from model ignorance directly informs feature reliability and risk-aware modeling. Furthermore, the contributions in this paper can be applied to estimating uncertainity features contribute to a model. These findings can be extended to help determine what features cause unreliability when making financial predictions under cetain market regimes. 
