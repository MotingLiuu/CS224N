# Assignment4 Self-Attention, Transformers, and Pretraining

## 1. Attention Exploration

$$
c = \sum_{i=1}^n v_i\alpha_i \\
\alpha_i = \frac{exp(k_i^Tq)}{\sum_{j=1}^iexp(k_j^Tq)}
$$

$alpha = \{ \alpha_1, ..., \alpha_n \}$is termed the "attention weights"

1. (a) for $\alpha_i = \frac{exp(k_i^Tq)}{\sum_{j=1}^iexp(k_j^Tq)}$ so when $k_j^Tq >> k_i^Tq$ where $i \in \{1, ..., n\} \ and\ i \neq j$

2. (b) $q = \frac{1}{2}(k_i + k_j)$
3. (c) **Drawbacks of single-headed attention**

   A set of key vectors $\{k_1, ..., k_n\}$are randomly sampled, $k_i \sim N(\mu_i, \Sigma_i)$, where the meas $\mu_i \in R^d$ are known. But the covariances $\Sigma_i$ are unknown. Further, assume that the means $\mu_i$ are perpendicular $\mu_i^T \mu_j = 0$ if $i\neq j$ and unit norm $||\mu_i|| = 1$

   1. $q = \frac{1}{2}(\mu_i + \mu_j) $
   2. One key vector $k_a$ may be larger or smaller in norm than the others, while still pointing in the same direction as $\mu_a$. Consider a covariance for item $a$ as $\sum_a = \alpha I + \frac{1}{2}(\mu_a \mu_a^T)$for vanishingly small $\alpha$. This casees $k_a$ to point in roughly the same direction as $\mu_a$but with large variances in magnitude. Further, let $\Sigma_i = \alpha I$for all $i \neq a$

      



