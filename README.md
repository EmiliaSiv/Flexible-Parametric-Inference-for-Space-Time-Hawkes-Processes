# Flexible Parametric Inference for Space-Time Hawkes Processes

## Install package under development 

To install the package under development, place yourself in the folder and run

```shell
pip install -e .
```

## Use

RUN python files:
- exp1_{kernel}.py: reproduces all results reported in the paper, subsection 4.1 and Appendix B, with kernel = {EXP, KUR, TG, POW_KUR}.
- exp1-plot_fig.py: reproduces Figures 1a, B1, B2.
- exp1-plot_fig_comptime.py: reproduces Figures 1b.
- exp1-plot_fig_params.py: reporduces Figures B3, B4.
- exp2_TG.py: reproduces all results reported in the paper, subsection 4.2.
- exp2-plot_fig.py: reproduces Figure 2a.
- exp2-plot_fig_comptime.py: reproduces Figure 2b.
- exp3_{kernel}.py: reproduces all results reported in the paper, Section 5 for real data, with kernel = {EXP, KUR, TG, POW_EXP, POW_KUR, POW_TG}.
- exp4-approx_psi.py: reproduces all results reported in the paper, subsection 4.3.

## Cite

If you use this code, please cite the corresponding work:

```bibtex
@misc{siviero2024flexibleparametricinferencespacetime,
      title={Flexible Parametric Inference for Space-Time Hawkes Processes}, 
      author={Emilia Siviero and Guillaume Staerman and Stephan Clémençon and Thomas Moreau},
      year={2024},
      eprint={2406.06849},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2406.06849}, 
}
```
