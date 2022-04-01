# Oil & Gas Equipment Failure Prediction

![](https://assets.spe.org/dims4/default/516db02/2147483647/strip/true/crop/1024x628+0+0/resize/800x491!/quality/90/?url=http%3A%2F%2Fspe-brightspot.s3.amazonaws.com%2F36%2Fb1%2F54da0536d608e68a8e8e8369b68f%2Fjpt-2020-05-29531hero.jpg)

## Intro
This assignment involves an FPSO (Floating Production, Storage, and Offloading) vessel, and we need to address an equipment failure problem.

An FPSO is a floating production system that receives fluids from a subsea reservoir through risers, which then separate fluids into crude oil, natural gas, water and impurities within the topsides production facilities onboard. Crude oil stored in the storage tanks of the FPSO is offloaded onto shuttle tankers to go to market or for further refining onshore.

To enable the operations of an FPSO, sensors are used to make sure the equipment does not fail. These sensors measure different parameters of the equipment in different setups configurations over time. We want to investigate one piece of equipment in different time cycles to understand what characteristics and parameters of the sensors might indicate that the equipment is on the verge of failing.

In summary, we need to create a data product that is able to make predictions about when the equipment will fail. This way, we can reduce costs with maintenance, and reduce delays caused by time-consuming unexpected stops.


## Roadmap

1) Data Cleaning
2) Exploratory Data Analysis
3) Feature Engineering & Feature Relevance
4) Model Selection & Model Evaluation
5) Hyperparameters Fine Tunning 
6) Evaluating the Reduction of Failures

## Final Product
Considering the previous analysis and tests, I'd deliver the following solution: use a combination of a regressor and a classifier so that the output would give the number of cycles before the equipment fails and the current probability of failure.

For the regressor, I'd select the GradientBoost regressor since it gave the lower RMSE. For the classifier, I'd choose the AdaBoost classifier since it gives better estimates of probabilities when compared to the NaiveBayes classifier.

The following lines give us an idea about how this model would work when it's in production.

![](https://i.ibb.co/Sc6ntjp/ndice.png)


The above graph shows us, as one could expect, that the number of cycles until failure drops down as the number of cycles increases. The dashed vertical lines indicate where failures actually occurred. Also, the probability of failure tends to increase whenever a failure gets closer. These results can be monitored so that they can support decision-makers to schedule maintenances and prevent unexpected stops.

Considering only the non-subsequent failures as unexpected failures, this solution reduces the total number of failures by 15%.

### Further Steps

As one further step, if more time were available, I'd propose creating regression models to predict the sensors' readings in the subsequent cycle of operation so that we can estimate the probability of failure in advance.

Besides, once this model gets into production, one should constantly update its training data so that the model may capture new causes of failures.
