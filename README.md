# fys-stk3155_proj1
Machine Learning UiO Project 1

## Reproducing the results

Use Python 3.14 and the project's virtual environment:

```sh
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/run_notebook.py
```

The runner starts a fresh kernel using that Python interpreter, executes every
code cell in order, saves `project1_runge.ipynb`, and exports the report figures
from the new notebook outputs. Nested cross-validation is the final evaluation;
the earlier fixed-split curves and full-data CV minima are exploratory/model
selection results, not independent final performance estimates. The final
Lasso comparison uses `LassoLars` with a checked stationarity residual;
the hand-written proximal solver is retained as an optimizer exercise.

Regression checks for batch-penalty normalization, Lasso stationarity, and
outer-fold isolation:

```sh
.venv/bin/python -m unittest discover -s tests
```
