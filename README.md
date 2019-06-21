# Case Study - Churn Prediction
 ![alt text](https://github.com/shillis17/ride_share_churn_predictions/blob/master/img/Ride_Shares.jpg)
__Question__: Which factors are the best predictors for retention regarding ride-share churn?

Table of Contents
<!--ts-->
 * [Cleaning](#cleaning)
 * [Model Comparison](#model-comparison)
 * [Metric Choice](#metric-choice)
 * [Final Model Results](#final-model-results)
 * [Conclusion](#conclusion)
<!--te-->

***
## Cleaning

The data didn't need much cleaning, we made a churn column based on whether or not the user's last ride was within the last 30 days from when the data was pulled (July 1st, 2014). So any person who's last ride was before June 1st, 2014 was considered churned. The date columns were dropped and string values were changed into dummy variables. All missing values were replaced with 0s.  The final features we predicted on were:
    
- avg_dist | Average ride distance
- avg_rating_by_driver | Average rating received from driver
- avg_rating_of_driver | Average rating given to driver
- avg_surge | Average surge multiplier on rides
- city | City (dummied for the three types)
- phone | Phone (dummied to either iPhone (1) or not iPhone (0))
- surge_pct | Percentage of rides taken with a surge multiplier
- trips_in_first_30_days | Rides taken in first 30 days
- luxury_car_user | Boolean on if the user took a luxury car in first 30 days
- weekday_pct | Percentage of rides that occur during weekdays


## Model Comparison
***
For our models we looked at a RandomForestClassifier and an XGBoostClassifier. We tuned the hyperparameters of these models using an extensive GridSearchCV. Overall the models came out relatively similar in performance. The confusion matrix for the RandomForest model was:

![rf_confusion_matrix](https://github.com/shillis17/ride_share_churn_predictions/blob/master/img/random_forest_confusion_matrix_normalized.png)

The matrix for the XGBoost model was:

![xg_confusion_matrix](https://github.com/shillis17/ride_share_churn_predictions/blob/master/img/gradient_boost_confusionmatrix.png)

These are really interesting to look at because the RandomForest model has a MUCH better true positive representation (66% vs 85% of true predictions) but the XGBoost model had a significantly better true negative representaion (86% vs 65% of false predictions)

There was much more correlation between the feature importance of the models though, as seen here:

![rf_feature_importance](https://github.com/shillis17/ride_share_churn_predictions/blob/master/img/random_forest_feature_importance.png)

![xg_feature_importance](https://github.com/shillis17/ride_share_churn_predictions/blob/master/img/gradient_boost_featureImp.png)

Both models considered the average rating received from the driver the most important feature with the type of phone also in the top 5. What's most interesting is that the boost model saw the luxury car use as a pretty important feature whereas the RandomForest model considered it the single least important.

So with these clear similarities and differences between the models, how did we decide which one to do with to test on the holdout data?

## Metric Choice
***
## Final Model Results
***
## Conclusion


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

Sunil Gupta of the Harvard Business School identifies "Lean Into Your Best Customers" as his number one strategy for retention. As such, we maximized our recall in order to cast the widest anti-churn net possible. We'd rather send out extra coupons, discounts, promos, etc. that won't be used than miss people who would maybe use the service an extra time due to a promotion.



What are the potential impacts of implementing these plans or decisions? What performance metrics did you use to evaluate these decisions, why?

If there is a tight marketing budget then it's better to maximize precision as ideally you want to try and prevent people who would churn first and foremost with the smallest investment possible.
