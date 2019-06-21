# Case Study - Churn Prediction

__Question__: Which factors are the best predictors for retention regarding ride-share churn?



# Work Flow
Perform any cleaning, exploratory analysis, and/or visualizations to use the provided data for this analysis.

Build a predictive model to help determine the probability that a rider will be retained.

Evaluate the model. Focus on metrics that are important for your statistical model.

Identify / interpret features that are the most influential in affecting your predictions.

Discuss the validity of your model. Issues such as leakage. For more on leakage, see this essay on Kaggle, and this paper: Leakage in Data Mining: Formulation, Detection, and Avoidance.

Repeat 2 - 5 until you have a satisfactory model.

Consider business decisions that your model may indicate are appropriate. Evaluate possible decisions with metrics that are appropriate for decision rules.


# Deliverables
Code you used to build the model. The more repeatable, self explanatory, the better.

A presentation including the following points:

How did you compute the target?


What model did you use in the end? We decided to go with XXXXXX. Why?

In the end, we went with the XG Boost model based upon an informed business decision requiring us to optimize our model for Recall.

Average rating by driver was the most influential factor in our analysis
- User experience was identified as the fourth most important factor in reducing churn
- While allowing users to rate their drivers may benefit an aggregate of other customers, a process whereby drivers rate riders does't benefit other users

Alternative models you considered? Why are they not good enough?
What performance metric did you use to evaluate the model? We used a confusion matrix to evaluate Recall, Accuracy... 

Why?
- Not as concerned about RMSE or R^2

We ran a Grid Search to compare metrics across the XGBoost and Random Forest Classifier.

Based on insights from the model, what plans do you propose to reduce churn?
Sunil Gupta of the Harvard Business School identifies "Lean Into Your Best Customers" as his number one strategy for retention. As such, we optimized 



What are the potential impacts of implementing these plans or decisions? What performance metrics did you use to evaluate these decisions, why?
