# conf-mc
Conformalized matrix completion

We propose a distribution-free method for predictive inference in the matrix completion problem. Our method adapts the framework of conformal prediction, which provides confidence intervals with guaranteed distribution-free validity in the setting of regression, to the problem of matrix completion.
Our resulting method, conformalized matrix completion (```cmc```), offers provable predictive coverage regardless of the accuracy of the low-rank model.
Empirical results on simulated and real data demonstrate that ```cmc``` is robust to model misspecification while matching the performance of existing model-based methods when the model is correct.

## Implementation
- In the folder ```python```, codes are available for simulations with homogeneous missingness (```conf-mc-synthetic.ipynb```), heterogeneous missingness (```conf-mc-hetero.ipynb```), and real dataset (```conf-mc-sales.ipynb```).
- In the folder ```plot```, figures in the paper can be replicated from simulation results stored in ```results```.
