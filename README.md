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

Sunil Gupta of the Harvard Business School identifies "Lean Into Your Best Customers" as his number one strategy for retention. As such, we maximized our recall in order to cast the widest anti-churn net possible. We'd rather send out extra coupons, discounts, promos, etc. that won't be used than miss people who would maybe use the service an extra time due to a promotion.

This led us to choosing the XGBoost model as our final model as it had the highest recall scores.

The main reason choosing recall could be the wrong choice is under a tight marketing budget. It’d be better to maximize precision as ideally you want to try and prevent people who would churn first and foremost with the smallest investment possible.


## Final Model Results
After reviewing the model results and deciding upon our final model we updated the model to use the entirety of the training set to train our model and used the holdout set to test, giving the model more data than the previous versions.

The final model output gaves us the following metrics.
Precision: 73.5
Recall: 65.88
Accuracy: 78.17

 ![alt text](https://github.com/shillis17/ride_share_churn_predictions/blob/master/img/final_model_confusionmatrix.png)


We also used this model to show the features of most importance based on thepredictions of the model.
 ![alt text](https://github.com/shillis17/ride_share_churn_predictions/blob/master/img/final_model_featureImp.png)
The top 3 features of importance based on this model were average rating by driver, the city the customer is located in and the type of phone they own
***
## Conclusion

The business could help itself by considering the following features toward improving churn:

- Given the most important on the feature list was average rating by driver in the context of churn, this decision by the company had all the potential to disincentivize customers--including its most loyal customers.
- Reorganizing incentives to loyal customers could combat this problem
- Onboarding customers more smoothly into the user process--e.g., via the app, could reduce churn, and simultaneously account for the feature importance of 'phone' which as a feature might not otherwise make sense as a prioritized feature
- Average surge: too many factors could inform the importance of this feature--too many to isolate (e.g., not enough drivers during a rush, a continuaton of an existing churn problem, a rebound pointing in favor of the company); churn could point downward toward average surge with weekday percentage; weekend promos could help that cause
- The average distance feature could be helped by promotions based on longer trips
***
# BIG SHOUT OUT TO JUPYTER NOTEBOOKS
