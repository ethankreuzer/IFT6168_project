# Sensitivity Analysis of Causal Discovery Methods - IFT6168 Project

This is the repository containing the associated code we used to perform our various experiments for our final project for IFT6168 --- Causality and Machine Learning.

## Overview

The extra datasets that we generated with the code from the DCDI repository are found in `generated_datsets/`.

The code and results of the baseline and increasing interventions experiments are available currently only on the `EthanH` branch.
The baseline experiment code is found in `clean_baseline_experiment.py` and associated notebooks.
The increasing number of interventional regimes experiment was conducted mainly via Shell scripts calling to the `main.py` function of the DCDI repository, with plotting conducted in `plot_ignoring_intervs_results.ipynb`.
The saved results of both of these experiments are found in `results/`.

## Acknowledgements
We acknowledge the following other repositories that we utilized in our work:
* [DCDI](https://github.com/slachapelle/dcdi);
* [DCD-FG](https://github.com/Genentech/dcdfg);
* [NOTEARS](https://github.com/xunzheng/notears).
