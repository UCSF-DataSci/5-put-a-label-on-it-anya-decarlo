# Assignment 5: Health Data Classification Results

This file contains your manual interpretations and analysis of the model results from the different parts of the assignment.

## Part 1: Logistic Regression on Imbalanced Data

### Interpretation of Results

- **Which metric performed best and why?**  
  The AUC score (0.8874) performed best because it evaluates the model's ability to distinguish between classes regardless of the specific threshold chosen, making it more robust to class imbalance. Unlike other metrics, AUC considers the ranking of predictions rather than just the binary outcomes.

- **Which metric performed worst and why?**  
  Precision (0.3229) performed worst due to the class imbalance in the dataset. With significantly more negative cases than positive ones, even a small number of false positives drastically reduces precision. This means only about 32% of positive predictions were actually correct.

- **How much did the class imbalance affect the results?**  
  Class imbalance significantly affected the results, as evidenced by the large gap between accuracy (0.8186) and F1 score (0.4593). The model achieves high accuracy by correctly classifying the abundant negative cases, but struggles with the minority positive class, which is more clinically important. The accuracy metric is misleading because of this imbalance.

- **What does the confusion matrix tell you about the model's predictions?**  
  The confusion matrix shows that while the model has good recall (113 out of 142 positive cases correctly identified, about 80%), it makes many false positive predictions (237 negative cases incorrectly classified as positive). This reveals that the model is biased toward predicting the positive class to compensate for imbalance, but at the cost of many false alarms.

## Part 2: Tree-Based Models with Time Series Features

### Comparison of Random Forest and XGBoost

- **Which model performed better according to AUC score?**  
  XGBoost performed better with an AUC score of 0.9964 compared to Random Forest's 0.9755. This represents a relative improvement of 2.13%, which is significant considering these already high AUC values.

- **Why might one model outperform the other on this dataset?**  
  XGBoost likely outperformed Random Forest because its gradient boosting approach excels at capturing complex patterns in time-series data. Unlike Random Forest which builds independent trees, XGBoost builds trees sequentially where each new tree corrects errors made by previous trees. This sequential learning is particularly advantageous for datasets with temporal dependencies like heart rate measurements over time. Additionally, XGBoost's regularization capabilities help prevent overfitting to noise in the physiological data.

- **How did the addition of time-series features (rolling mean and standard deviation) affect model performance?**  
  The addition of time-series features substantially improved model performance for both algorithms, as evidenced by the high AUC scores (0.9755 and 0.9964) compared to the basic logistic regression model from Part 1 (0.8874). Rolling statistics captured trends and variability in heart rate patterns over time, providing critical information about cardiovascular dynamics that static snapshots miss. These temporal features allowed the models to identify subtle patterns preceding health events, demonstrating the importance of considering how physiological measurements change over time rather than just their absolute values.

## Part 3: Logistic Regression with Balanced Data

### Improvement Analysis

- **Which metrics showed the most significant improvement?**  
  Comparing Part 3 (balanced) with Part 1 (imbalanced), the F1 score showed the most significant improvement, increasing from 0.4593 to 0.5289 (a 15% improvement). Precision also improved notably from 0.3229 to 0.3864 (a 19.7% relative increase). These improvements are critical because they represent better balance between precision and recall.

- **Which metrics showed the least improvement?**  
  Recall showed the least improvement, increasing only modestly from 0.7958 to 0.8380 (a 5.3% improvement). This suggests that the original model was already fairly good at identifying positive cases, but was making too many false positive predictions.

- **Why might some metrics improve more than others?**  
  Precision and F1 improved more significantly because SMOTE directly addresses the fundamental problem of class imbalance by generating synthetic examples of the minority class. This helps the model learn a more balanced decision boundary, reducing false positives without sacrificing much in terms of recall. Accuracy and AUC showed moderate improvements because they're already somewhat robust to class imbalance. Recall improved less because it was already high in the imbalanced model, which tended to over-predict the positive class.

- **What does this tell you about the importance of addressing class imbalance?**  
  Addressing class imbalance is crucial for developing clinically useful models, particularly when false positives and false negatives have different consequences. The improvements across all metrics demonstrate that balanced training data leads to more reliable and trustworthy predictions. Without addressing class imbalance, models may appear to perform well on aggregate metrics like accuracy, but fail on the critical task of correctly identifying the minority class without excessive false alarms. For health data classification specifically, this could mean the difference between a usable diagnostic tool and one that generates too many false positives to be practical.

## Overall Conclusions

- **What were the most important factors affecting model performance?**  
  The most important factors affecting model performance were: (1) Class imbalance in the dataset, which significantly impacted precision and F1 scores; (2) The incorporation of time-series features from heart rate data, which dramatically improved the model's predictive power as evidenced by the jump in AUC scores; and (3) Model selection, with more sophisticated ensemble methods like XGBoost outperforming simpler models. The temporal nature of the health data also played a crucial role, with models that could effectively capture trends over time performing substantially better than those working with static snapshots.

- **Which techniques provided the most significant improvements?**  
  The technique that provided the most dramatic improvement was the addition of time-series features (rolling mean and standard deviation of heart rate), which elevated AUC scores from around 0.89 to over 0.99 with XGBoost. This was followed by the choice of advanced tree-based models over logistic regression. SMOTE for addressing class imbalance provided more modest but still important improvements, particularly in precision and F1 score, making the model more balanced in its predictive capabilities. The categorical encoding of features like smoker_status also contributed to improved performance in the final model.

- **What would you recommend for future modeling of this dataset?**  
  For future modeling of this health dataset, I would recommend: (1) Further exploration of time-series features beyond rolling statistics, such as rate of change, frequency domain features, or anomaly detection metrics; (2) Implementing more sophisticated class balancing techniques, including adaptive sampling methods or cost-sensitive learning; (3) Exploring deep learning approaches like RNNs or LSTMs that can inherently model temporal dependencies; (4) Incorporating feature importance analysis to identify the most predictive variables and potentially reduce model complexity; (5) Implementing threshold optimization to balance precision and recall based on the specific clinical context; and (6) Developing ensemble methods that combine the strengths of multiple models. Finally, I would recommend exploring interpretability techniques to make the models more transparent and trustworthy for healthcare applications.