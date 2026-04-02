# 1. Algorithm Summary

K-Nearest Neighbors (KNN) is an instance-based supervised learning algorithm that predicts labels by comparing a new sample with previously observed training samples in feature space. It is a lazy learner, meaning there is no explicit parametric training phase that produces a compact set of learned coefficients. Instead, the model stores the training data and performs prediction at inference time by identifying the nearest neighbors according to a distance function and assigning the class through neighbor voting. In this project, distance-based reasoning is central because cancellation behavior is inferred from local similarity patterns rather than a fixed global decision equation.

# 2. Why KNN for This Dataset

The hotel booking dataset contains heterogeneous tabular features and likely non-linear relationships between booking attributes and cancellation outcomes. KNN is suitable in this context because it can represent non-linear decision boundaries without imposing strong assumptions about the underlying data distribution. Unlike linear models that rely on global linear separability, KNN adapts locally to the structure of the data. It was therefore used as a strong baseline model to test how far neighborhood-based classification could perform under the shared preprocessing pipeline and evaluation protocol.

# 3. Important Design Choices

The final KNN implementation used a fully integrated preprocessing-and-model pipeline with missing-value imputation, quantile clipping (1%-99%), one-hot encoding, and standard scaling. Because KNN depends on distance calculations, the pipeline was configured to output dense transformed features before model fitting.

Hyperparameter tuning was conducted with stratified 5-fold cross-validation and F1-score as the optimization objective. The selected best configuration was `n_neighbors = 15`, `weights = distance`, and `metric = manhattan`.

## 3.1 Distance Metric

Two distance metrics were evaluated: Euclidean and Manhattan. In high-dimensional feature spaces, especially after one-hot encoding, Euclidean distances can become less discriminative due to distance concentration effects associated with the curse of dimensionality. Manhattan distance is often more stable in such settings because it aggregates coordinate-wise absolute differences and can preserve relative neighborhood structure more effectively when many sparse/binary indicator dimensions are present. The cross-validation results supported this behavior, with Manhattan selected as the best-performing metric.

## 3.2 Number of Neighbors (k)

The value of k controls the bias-variance tradeoff in KNN. Small k values can overfit local noise and produce high-variance decisions, while very large k values oversmooth local structure and increase bias. Through cross-validated model selection, `k = 15` provided the best balance between variance reduction and preservation of class-discriminative locality, yielding the strongest F1-score among tested settings.

## 3.3 Weighting Strategy

Two voting strategies were compared: uniform weighting and distance weighting. Uniform weighting treats all neighbors equally, regardless of their distance from the query point. Distance weighting gives larger influence to closer neighbors, which is often beneficial when local neighborhoods contain mixed classes near decision boundaries. The best model used `weights = distance`, indicating that proximity-aware voting improved predictive quality for cancellation classification.

# 4. Class Imbalance Handling

This task is class-imbalanced (approximately 72% not canceled vs 28% canceled), so accuracy alone is potentially misleading because a model can achieve high accuracy while under-detecting cancellations. For that reason, model development prioritized F1-score as the primary optimization metric, and additional discrimination analysis used precision-recall behavior rather than relying only on aggregate accuracy.

KNN does not provide a native `class_weight` parameter, so imbalance mitigation was handled through objective and decision calibration choices: stratified CV with F1 optimization, precision-recall evaluation, and threshold tuning on predicted probabilities. Threshold analysis showed that lowering the decision threshold from 0.50 to 0.40 improved F1-score, primarily by increasing recall at the cost of some precision. This explicitly reflects the precision-recall tradeoff required in cancellation detection, where missing true cancellations can be costly.

# 5. Model Performance Summary

The final test-set performance is summarized below:

| Metric            |  Value |
| ----------------- | -----: |
| Accuracy          | 0.8100 |
| Balanced Accuracy | 0.7449 |
| Precision         | 0.6732 |
| Recall            | 0.6002 |
| F1-score          | 0.6346 |
| ROC-AUC           | 0.8506 |
| PR-AUC            | 0.6967 |
| Log Loss          | 0.6315 |

These results indicate good ranking quality and class separability (ROC-AUC 0.8506), while recall remains moderate, implying that a non-trivial fraction of cancellations is still missed. Precision is higher than recall in the default operating point, consistent with a conservative positive prediction behavior. Threshold tuning partially addresses this by trading precision for improved recall and F1-score.

# 6. Limitations of KNN

KNN has several important operational and statistical limitations. First, prediction is computationally expensive because inference requires distance computation against stored training instances, leading to approximately O(n) query-time complexity per sample (before indexing optimizations). Second, memory usage is high because the model retains the full training dataset rather than a compact parameter set. Third, KNN is sensitive to irrelevant and noisy features, which can distort neighborhood quality if feature engineering is not carefully controlled. Fourth, performance can degrade in high-dimensional spaces, where distance metrics become less informative and neighborhood structure weakens. Finally, because KNN is distance-based, proper scaling is mandatory; unscaled numeric ranges can dominate distance calculations and bias predictions.

# 7. Improvements & Future Work

Several improvements are practical for future iterations. Feature selection can reduce noisy or redundant dimensions and improve neighborhood quality. Dimensionality reduction methods such as PCA may improve both efficiency and generalization in high-dimensional encoded spaces. Approximate nearest neighbor search structures can reduce inference latency and make KNN more deployable at scale. Beyond pure KNN, hybrid pipelines and ensemble strategies could improve cancellation detection by combining local similarity modeling with more expressive learners that better capture global structure.
