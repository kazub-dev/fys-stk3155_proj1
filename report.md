# Polynomial Regression for the Runge Function

**Author:** Katarzyna Anna Zubowicz  
**Course:** FYS-STK3155  
**Project:** Project 1

## Abstract

In this project I studied polynomial regression for the Runge function

$$
f(x)=\frac{1}{1+25x^2}, \qquad x\in[-1,1],
$$

with additive Gaussian noise. I implemented ordinary least squares (OLS),
Ridge regression, bootstrap resampling, cross-validation, gradient-based
optimizers, Lasso regression, and stochastic gradient descent. The main data
set contained 100 equally spaced points, noise standard deviation
\(\sigma=0.1\), and polynomial degrees from 0 to 15. The data were split into
80% training and 20% test data.

For the fixed exploratory split, OLS had its lowest printed test MSE at degree
12, with test MSE \(0.006079\) and test \(R^2=0.933768\). Ridge achieved a
slightly lower test MSE of \(0.006055\) with \(\lambda=10^{-4}\) and degree
12. In the final 10-fold cross-validation, the best OLS model had degree 12
and MSE \(0.013525749405278602\), while the best Ridge model had
\(\lambda=10^{-4}\), degree 12, and MSE \(0.013309094857936087\). The best
Lasso model in that final comparison had \(\lambda=10^{-5}\), degree 6, and
MSE \(0.02926334238976904\). The experiments also showed that increasing the
polynomial degree reduces training error but can increase test error, and that
regularization and resampling are useful for assessing this tradeoff.

## 1. Introduction

### 1.1 Motivation and problem

The purpose of this project was to study several supervised learning methods
and resampling techniques on a controlled regression problem. I used the Runge
function because it is smooth but difficult to approximate well with high-order
polynomials near the ends of the interval. This makes it suitable for
studying underfitting, overfitting, regularization, and the bias--variance
tradeoff.

The project required comparisons between OLS, Ridge, and Lasso regression,
together with bootstrap and cross-validation methods. I also investigated
gradient descent and several adaptive optimization methods, and compared their
solutions with the closed-form OLS and Ridge solutions.

### 1.2 Structure of the report

Section 2 introduces the regression models, error measures, resampling
methods, and optimization methods. Section 3 describes the data generation,
scaling, splitting, and experimental setup. Section 4 describes the
implementation and numerical checks. Section 5 presents the stored numerical
results and their interpretation. Section 6 discusses limitations and critical
points, and Section 7 summarizes the main conclusions.

## 2. Theory and formalism

### 2.1 Data model and polynomial design matrix

I generated observations according to

$$
y_i=f(x_i)+\epsilon_i,\qquad \epsilon_i\sim\mathcal{N}(0,\sigma^2).
$$

The polynomial model of degree \(p\) was represented by

$$
\tilde y_i=\sum_{j=0}^{p}\theta_j x_i^j
  =\boldsymbol{x}_i\boldsymbol{\theta}.
$$

The corresponding design matrix contains the columns
\(1,x,x^2,\ldots,x^p\). In the implementation, the input coordinate was
standardized using the training data before constructing this matrix.

### 2.2 MSE and \(R^2\)

I used the mean squared error

$$
\operatorname{MSE}(\boldsymbol y,\tilde{\boldsymbol y})
=\frac{1}{n}\sum_{i=0}^{n-1}(y_i-\tilde y_i)^2
$$

and the coefficient of determination

$$
R^2
=1-\frac{\sum_i(y_i-\tilde y_i)^2}
{\sum_i(y_i-\bar y)^2}.
$$

The MSE was used to select models in the cross-validation experiments.

### 2.3 OLS regression

For OLS, the cost function is the mean squared error. The coefficient estimate
can be computed with the pseudoinverse:

$$
\hat{\boldsymbol\theta}
=\boldsymbol X^+\boldsymbol y.
$$

I used `numpy.linalg.pinv` rather than explicitly inverting
\(\boldsymbol X^T\boldsymbol X\). This is a numerically safer implementation of
the same least-squares solution for the design matrices used here.

### 2.4 Ridge regression

Ridge regression adds an \(\ell_2\) penalty to the coefficients. The
implementation used a penalty matrix whose intercept entry was zero, so that
the intercept was not penalized:

$$
\hat{\boldsymbol\theta}_{\mathrm{Ridge}}
=
(\boldsymbol X^T\boldsymbol X+\lambda\boldsymbol P)^+
\boldsymbol X^T\boldsymbol y.
$$

Here \(\lambda\geq0\), and \(\boldsymbol P\) is the identity matrix except for
its first diagonal entry, which is zero. In the singular-value interpretation,
regularization reduces the contribution from modes with small singular values.

### 2.5 Bias--variance decomposition and bootstrap

For data generated as \(y=f(x)+\epsilon\), the expected prediction error can
be decomposed as

$$
\mathbb E[(y-\tilde y)^2]
=\operatorname{Bias}^2[\tilde y]
+\operatorname{Var}[\tilde y]
+\sigma^2.
$$

The bias term measures the squared difference between the true function and
the average model prediction. The variance term measures the variation of
model predictions over different training sets. The final term is the noise
variance.

I estimated these quantities with bootstrap resampling of the training data.
For each degree, I generated 200 bootstrap samples, fitted an OLS polynomial
to each sample, and evaluated predictions on a grid. The stored implementation
uses the known noise-free Runge function for the bias calculation, so the bias
estimate is not obtained by replacing the unknown function with noisy
observations.

### 2.6 Cross-validation

I implemented shuffled 5-fold and 10-fold cross-validation. For every fold,
the scaling parameters were calculated from the fold's training data and then
applied to the corresponding validation data. This avoids allowing validation
data to influence preprocessing.

Cross-validation was used to estimate the MSE as a function of polynomial
degree for OLS and as a function of degree and \(\lambda\) for Ridge. A final
10-fold comparison also included Lasso.

### 2.7 Gradient descent and automatic differentiation

For OLS, the analytical gradient used was

$$
\nabla_{\boldsymbol\theta}C
=\frac{2}{n}\boldsymbol X^T
(\boldsymbol X\boldsymbol\theta-\boldsymbol y).
$$

For Ridge, the gradient contains the corresponding penalty contribution for
the non-intercept coefficients. Gradient descent updates the parameters as

$$
\boldsymbol\theta_{k+1}
=\boldsymbol\theta_k-\eta\nabla C(\boldsymbol\theta_k).
$$

I also used JAX to calculate gradients automatically. Automatic
differentiation applies the chain rule to the operations in the cost function;
it is not symbolic differentiation and it is not finite-difference
approximation.

The plain gradient-descent stability limit is connected to the largest
eigenvalue of the Hessian. For the OLS Hessian, the implementation calculated
\(\eta_{\max}=2/\lambda_{\max}(H)\).

### 2.8 Momentum, AdaGrad, RMSprop, and Adam

I compared ordinary gradient descent with momentum, AdaGrad, RMSprop, and
Adam. These methods modify the update by using previous gradients or
coordinate-dependent learning-rate factors. Their purpose in this project was
to compare convergence speed and accuracy relative to the closed-form Ridge
solution.

### 2.9 Lasso and proximal updates

Lasso adds an \(\ell_1\) penalty. Because the absolute-value function is not
differentiable at zero, I used a proximal soft-thresholding update for the
non-intercept coefficients:

$$
\operatorname{soft}(z,t)
=\operatorname{sign}(z)\max(|z|-t,0).
$$

The derivative returned by JAX for \(|\theta|\) at zero was `1.0`. This is a
valid member of the subgradient interval \([-1,1]\), although it is not the
only valid subgradient at zero.

## 3. Data and experimental setup

The default configuration in the solution notebook was:

| Quantity | Value |
|---|---:|
| Number of points | 100 |
| Domain | \([-1,1]\) |
| Noise standard deviation | 0.1 |
| Test fraction | 0.2 |
| Maximum polynomial degree | 15 |
| Default seed | 2026 |

The points were equally spaced in \([-1,1]\). The noisy response was generated
with a NumPy random generator. I used `train_test_split` with
`random_state=2026`.

Before constructing the design matrices, I standardized \(x\) using the mean
and standard deviation calculated from the training data only. The same
training-data transformation was applied to the test data. This prevents the
test inputs from determining the transformation.

The experiments varied the following:

- Polynomial degree from 0 to 15.
- Number of data points: 20, 50, 150, 250, and 500.
- Noise standard deviation: 0.0, 0.01, 0.05, 0.2, and 0.5.
- Ten random seeds for a test-error variability plot.
- Ridge penalties \(0\), \(10^{-6}\), \(10^{-4}\), \(10^{-2}\), \(0.1\), and
  \(1.0\).
- Five-fold and ten-fold cross-validation.
- 200 bootstrap resamples.
- Mini-batch sizes 1, 8, 16, 40, and the complete training set for the SGD
  experiment.

The final cross-validation used degrees 0--15 and Lasso/Ridge penalties
\(10^{-5}\), \(10^{-4}\), \(10^{-3}\), and \(10^{-2}\).

## 4. Implementation, testing, and numerical checks

The notebook defines reusable functions for data generation, scaling, design
matrix construction, prediction, MSE, \(R^2\), OLS, Ridge, bootstrap,
cross-validation, gradient descent, proximal Lasso, and mini-batch SGD.
Results are plotted with Matplotlib, and numerical summaries are printed for
the main comparisons.

The OLS solver uses the pseudoinverse. The Ridge solver uses the penalized
normal equations and excludes the intercept from the penalty. The bootstrap
and cross-validation routines refit models for each resampled or folded
training set.

I checked the gradient implementations against JAX. The source uses the same
normalized Ridge penalty convention in the closed-form solver, cost function,
analytical gradient, automatic-differentiation gradient, and iterative
optimizers. The affected notebook cells were rerun after this correction.

For plain gradient descent on degree 5, the reported coefficient errors
relative to the closed-form solutions were \(1.087\times10^{-7}\) for OLS and
\(1.082\times10^{-7}\) for Ridge, after 18012 and 17924 iterations,
respectively.

The notebook reports \(\eta_{\max}=0.8937721839600539\) for OLS and
\(\eta_{\max}=0.8937623265903589\) for Ridge for the gradient-descent test
problem.

## 5. Results and analysis

### 5.1 OLS as a function of degree

The fixed-split OLS results were:

| Degree | Train MSE | Test MSE | Train \(R^2\) | Test \(R^2\) |
|---:|---:|---:|---:|---:|
| 0 | 0.110524 | 0.098739 | 0.000000 | -0.075725 |
| 1 | 0.110470 | 0.098486 | 0.000485 | -0.072965 |
| 2 | 0.060390 | 0.036142 | 0.453600 | 0.606251 |
| 3 | 0.060388 | 0.036081 | 0.453618 | 0.606914 |
| 4 | 0.038588 | 0.017857 | 0.650861 | 0.805457 |
| 5 | 0.038487 | 0.017814 | 0.651774 | 0.805919 |
| 6 | 0.025386 | 0.010585 | 0.770313 | 0.884679 |
| 7 | 0.024668 | 0.013008 | 0.776812 | 0.858278 |
| 8 | 0.018443 | 0.012482 | 0.833134 | 0.864016 |
| 9 | 0.018342 | 0.012189 | 0.834041 | 0.867209 |
| 10 | 0.012719 | 0.009696 | 0.884925 | 0.894367 |
| 11 | 0.012494 | 0.011605 | 0.886953 | 0.873566 |
| 12 | 0.011010 | 0.006079 | 0.900384 | 0.933768 |
| 13 | 0.010920 | 0.007445 | 0.901194 | 0.918892 |
| 14 | 0.008591 | 0.039395 | 0.922273 | 0.570807 |
| 15 | 0.008423 | 0.061169 | 0.923791 | 0.333587 |

Training MSE decreases almost monotonically as the degree increases. Test MSE
does not: it is lowest at degree 12 and then rises strongly at degrees 14 and
15. The corresponding decline in test \(R^2\) shows the effect of
overfitting. In contrast, degrees 0 and 1 underfit the curved Runge function,
as indicated by their negative test \(R^2\) values.

The notebook includes plots of MSE and \(R^2\) versus degree, fitted
polynomials for degrees 1, 3, 7, 10, 12, and 15, and the OLS parameters as
the degree increases. Figures 1--3 show the stored OLS plots for the fixed
split, with \(n=100\), \(\sigma=0.1\), and seed 2026.

![OLS training and test MSE versus polynomial degree.](report_figures/ols_mse.png)

*Figure 1: Stored OLS training and test MSE curves for the fixed 80/20 split.
The additional “Actual” curve is evaluated against the noiseless Runge values
on the test points.*

![OLS training and test R-squared versus polynomial degree.](report_figures/ols_r2.png)

*Figure 2: Stored OLS training and test \(R^2\) curves for the fixed split.*

![Polynomial fits for selected degrees.](report_figures/ols_fits.png)

*Figure 3: Stored polynomial fits for selected degrees, together with the
training and test observations and the noise-free Runge function.*

### 5.2 Ridge regression

The best fixed-split test results for each penalty were:

| \(\lambda\) | Best degree | Test MSE | Test \(R^2\) |
|---:|---:|---:|---:|
| \(0.0\) | 12 | 0.006079 | 0.933768 |
| \(10^{-6}\) | 12 | 0.006079 | 0.933775 |
| \(10^{-4}\) | 12 | 0.006055 | 0.934032 |
| \(10^{-2}\) | 10 | 0.007368 | 0.919731 |
| \(0.1\) | 12 | 0.008835 | 0.903742 |
| \(1.0\) | 8 | 0.011464 | 0.875109 |

The very small Ridge penalty \(10^{-4}\) slightly improved the best test MSE
relative to OLS. Larger penalties changed the best degree and produced larger
minimum test errors in this experiment. The stored notebook also plots the
singular-mode shrinkage factor

$$
\frac{s^2}{s^2+\lambda},
$$

for the degree-15 design matrix. This gives a direct numerical illustration
that Ridge suppresses modes more strongly when their singular values are
small or when \(\lambda\) is increased.

![Ridge test MSE and test R-squared for several penalties.](report_figures/ridge_test_metrics.png)

*Figure 4: Stored Ridge test MSE and test \(R^2\) curves as functions of
polynomial degree for the listed penalty values.*

### 5.3 Bootstrap and bias--variance results

The bootstrap experiment used 200 training-set resamples for every polynomial
degree. The implementation calculates held-out test MSE on `x_test` and
`y_test`, while storing the noiseless-grid error separately as `grid_mse`.
After rerunning the affected cells, the lowest held-out bootstrap prediction
MSE occurred at degree 4.

The notebook plots bootstrap training MSE and prediction MSE, together with
the estimated squared bias, variance, and noise level. These curves show the
intended relationship between increasing model complexity, decreasing bias,
and increasing variance. The exact numerical values of every plotted
bootstrap curve were not printed in the notebook, so I do not report
additional point values here.

![Bootstrap prediction error and bias--variance decomposition.](report_figures/bootstrap_bias_variance.png)

*Figure 5: Stored bootstrap training/prediction curves and estimated
bias--variance components. The corrected held-out metric must be regenerated
before this figure is used as quantitative evidence.*

### 5.4 Cross-validation

The stored cross-validation results gave:

| Cross-validation | OLS best degree | Ridge best \(\lambda\) | Ridge best degree |
|---|---:|---:|---:|
| 5-fold | 12 | \(10^{-6}\) | 12 |
| 10-fold | 12 | \(10^{-4}\) | 12 |

Both fold choices selected degree 12 for OLS and Ridge. The difference between
the preferred Ridge penalties shows that the penalty selection is somewhat
sensitive to the fold partition for this data set, even though the selected
degree is stable.

![Five-fold and ten-fold cross-validation curves.](report_figures/cross_validation.png)

*Figure 6: Stored 5-fold and 10-fold cross-validation curves for OLS and the
shown Ridge penalty values.*

### 5.5 Gradient descent and adaptive optimizers

For the Ridge degree-5 problem, the stored optimizer comparison was:

| Method | Iterations | Coefficient error |
|---|---:|---:|
| Plain | 20000 | \(8.754\times10^{-2}\) |
| Momentum | 19282 | \(1.904\times10^{-6}\) |
| AdaGrad | 6962 | \(6.052\times10^{-7}\) |
| RMSprop | 20000 | \(1.225\times10^{-2}\) |
| Adam | 692 | \(2.393\times10^{-8}\) |

In this particular parameter setting, Adam reached the smallest coefficient
error in the fewest iterations. AdaGrad and momentum also approached the
closed-form solution closely. Plain gradient descent and RMSprop reached the
iteration limit before meeting the chosen stopping criterion, so their larger
errors should not be interpreted as evidence that their limiting solutions
are different from the Ridge solution. An additional Adam learning-rate sweep
was rerun and is stored in the notebook.

### 5.6 Lasso

The proximal Lasso experiment used \(\lambda=10^{-3}\). The stored output
reported:

| Quantity | Value |
|---|---:|
| Iterations | 18304 |
| Nonzero non-intercept coefficients | 5 |
| Training MSE | 0.0385160062400922 |
| JAX derivative of `abs` at zero | 1.0 |

The five nonzero non-intercept coefficients demonstrate the sparsifying
effect of the soft-thresholding step in this experiment. The Lasso result
cannot be compared directly to the OLS degree-5 training MSE without also
considering the penalty and the different fitted coefficients.

### 5.7 Stochastic gradient descent

The stored SGD experiment used Ridge with \(\lambda=10^{-2}\), batch size 16,
1000 epochs, and learning rate 0.005. It reported:

| Quantity | Value |
|---|---:|
| Coefficient error relative to closed-form Ridge | 0.05757345959790696 |
| Training MSE | 0.03869627311608938 |

The notebook also plotted coefficient error as a function of mini-batch size
for batch sizes 1, 8, 16, 40, and the complete training set. The saved notebook
also reports an epoch/learning-rate comparison at batch size 16: the coefficient
errors were \(0.7125\), \(0.2167\), and \(0.05757\) for 100, 500, and 1000
epochs at learning rate \(0.005\), and \(0.5279\) for 1000 epochs at learning
rate \(0.001\). The notebook does not print the individual values from the
mini-batch-size plot, so I do not replace that visual comparison with
unreported numerical values.

### 5.8 Final OLS, Ridge, and Lasso comparison

The final comparison used 10-fold cross-validation and the same generated
data configuration with \(n=100\), \(\sigma=0.1\), and seed 2026:

| Method | Selected hyperparameters | Cross-validated MSE |
|---|---|---:|
| OLS | Degree 12 | 0.013525749405278602 |
| Ridge | \(\lambda=0.0001\), degree 12 | 0.013309094857936087 |
| Lasso | \(\lambda=1e-05\), degree 6 | 0.02926334238976904 |

Ridge gave the lowest cross-validated MSE among the three final models, but
only slightly lower than OLS. Lasso selected a lower degree and had a larger
cross-validated MSE in this comparison. This suggests that the sparsity
induced by Lasso was not as effective as the mild coefficient shrinkage from
Ridge for this particular Runge-function experiment.

![Final cross-validation comparison of OLS, Ridge, and Lasso.](report_figures/final_cv.png)

*Figure 7: Stored final 10-fold cross-validation MSE curves for OLS, Ridge,
and Lasso across polynomial degrees and the displayed penalty values.*

## 6. Discussion and limitations

The results illustrate the central complexity tradeoff. Low-degree
polynomials have too little flexibility to represent the Runge function,
while very high-degree polynomials fit the training data increasingly well but
perform worse on held-out data. The best degree depends on how the error is
estimated: the fixed split and cross-validation both selected degree 12 for
the main OLS/Ridge comparisons, while the bootstrap prediction-MSE curve
selected degree 6.

Scaling the input using only training data is appropriate because it avoids
using held-out information during preprocessing. It also makes the polynomial
design matrix better behaved numerically than using unscaled powers directly.
The reported coefficient values therefore belong to the scaled-coordinate
polynomial basis.

The Ridge results support the expected interpretation of regularization:
small penalties can reduce variance without greatly changing the fit, whereas
larger penalties shrink the coefficients more strongly and can increase bias.
The final cross-validation results show that mild Ridge regularization was
slightly better than OLS for this data set.

The optimizer comparison shows that convergence depends strongly on the
learning-rate and update rule. The closed-form OLS and Ridge solutions provide
useful reference points for testing iterative methods. In particular, the
stored results show that Adam reached a small coefficient error much faster
than the plain method for the selected settings. This conclusion is specific
to the reported learning rates and stopping criteria.

There are also limitations in the numerical evidence. The notebook uses one
main 80/20 exploratory split and then uses cross-validation for model
selection; it does not report a separate final validation set. Some
experiments are represented by figures without printed numerical tables, so
their exact plotted values cannot be reproduced from the text alone. The
the stored outputs now correspond to the corrected objective and metric
implementations. Finally, the stored experiments
do not provide a systematic timing benchmark for computational cost, even
though the project asks for discussion of computational efficiency.

## 7. Conclusions and future work

I implemented and compared OLS, Ridge, and Lasso regression for noisy
observations of the Runge function. OLS and Ridge both selected degree 12 in
the main cross-validation analyses. Ridge with \(\lambda=10^{-4}\) achieved
the lowest final cross-validated MSE,
\(0.013309094857936087\), while OLS obtained
\(0.013525749405278602\). Lasso selected degree 6 with
\(\lambda=10^{-5}\), but its final MSE was higher,
\(0.02926334238976904\).

The experiments demonstrated that training error alone is not sufficient for
selecting polynomial degree. Test errors, bootstrap estimates, and
cross-validation reveal the effects of bias and variance. Ridge regularization
was useful here because a small penalty improved the held-out result without
substantially changing the selected degree. The iterative methods could
approach the closed-form solutions, but their performance depended on the
learning-rate schedule and optimizer.

Future work could include a systematic learning-rate and epoch study, timing
measurements for the closed-form and iterative methods, repeated nested
cross-validation, and a more complete numerical comparison of the bootstrap
and cross-validation estimates. These would make the conclusions less
dependent on a single split and would provide a stronger computational
benchmark.

## Appendix: Use of AI/LLM Tools

### Tools used

- ChatGPT, accessed 28 September--4 October 2026.

### Text writing and editing

The report was drafted from the project notebook and the course documents with
ChatGPT assistance. The contribution levels below describe this report-writing
interaction; the report must be read, understood, and endorsed by me before
submission.

| Report section | LLM contribution level | Notes |
|---|---:|---|
| Abstract | 3 | Drafted from the notebook's stored results; numerical claims were kept as printed. |
| Introduction | 3 | Drafted from the assignment and solution scope. |
| Theory and formalism | 3 | Drafted from the methods present in the notebook and assignment. |
| Data and experimental setup | 3 | Drafted from configuration values and code. |
| Implementation, testing, and numerical checks | 3 | Drafted from code and saved outputs, including the Ridge normalization discrepancy. |
| Results and analysis | 3 | Drafted from saved printed results and descriptions of stored plots. |
| Discussion and limitations | 3 | Drafted only from issues visible in the files and stored outputs. |
| Conclusions and future work | 3 | Drafted from the reported comparisons and supported limitations. |
| AI/LLM appendix | 3 | Structured to follow `guidelines.md`. |
| References | 2 | Formatted from sources named in `Project1.ipynb`; source material was checked during the project. |

For the project itself, ChatGPT was used for research and debugging during
28 September--4 October 2026. It did not substantially write or rewrite the
submitted code. I verified its assistance by running the code and checking the
source material used for the research. The model choices, experiments,
numerical results, and final scientific interpretation are my responsibility.

After reviewing the completed notebook, GitHub Copilot acted as a debugging and
review aid to identify and correct three groups of issues. GitHub Copilot made
the changes in `project1_runge.ipynb`: it made the Ridge penalty convention
consistent across the cost, analytical gradient, automatic-differentiation
check, and iterative methods; separated held-out bootstrap test MSE from
noiseless-grid prediction error and aligned the SGD penalty with the Ridge
objective; and connected the gradient-descent routine to its analytical and
automatic-differentiation gradient functions. It also added the notebook
disclosure required by the course guidelines. The resulting notebook
structure and JSON validity were checked, but the old saved outputs were not
treated as regenerated results.

### Code generation and assistance

| File / notebook | LLM level | Description |
|---|---:|---|
| `project1_runge.ipynb` | 1 | ChatGPT helped identify and debug errors in several parts of the code; it did not write or substantially rewrite the code. |

The notebook contains a Markdown disclosure immediately before the
gradient-descent implementation. The assistance was debugging-level rather
than code-generation-level, so no generated function or code skeleton is
claimed.

## References

1. FYS-STK3155/FYS4155 course lecture notes, especially Chapters 2--4:
   training and test error, bias--variance tradeoff, bootstrap,
   cross-validation, linear regression, regularization, scaling, and gradient
   descent.
2. Wessel N. van Wieringen, *Lecture Notes on Ridge Regression*,
   arXiv:1509.09169, https://arxiv.org/abs/1509.09169.
3. Trevor Hastie, Robert Tibshirani, and Jerome H. Friedman, *The Elements of
   Statistical Learning*, Springer,
   https://www.springer.com/gp/book/9780387848570.
4. Atılım Baydin, Barak A. Pearlmutter, Alexey Andreyevich Radul, and Jeffrey
   Mark Siskind, “Automatic Differentiation in Machine Learning: a Survey,”
   *Journal of Machine Learning Research* 18 (2018),
   https://jmlr.org/papers/v18/17-468.html.
5. JAX documentation, https://jax.readthedocs.io.
6. Scikit-learn documentation, https://scikit-learn.org.
7. Runge phenomenon overview,
   https://en.wikipedia.org/wiki/Runge%27s_phenomenon.
