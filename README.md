# ML-linearRegresion-DemandProjection

Linear regression model was carried out with the use of Machine Learning (ML) to demand forecast the following six (6) months from a xlx file sales and inventory data. Data from excel file has more than 280 different products from different categories.


Code Documentation in Spanish Step by step
Data imported from xlsx file, results exported to xlsx file
Forecasting using python pandas, numpy, matplotlib, sklearn.
*Included data cleansing steps


In the first place, an exploration of the data was carried out to perform an efficient analysis and processing of the data.
Using statistical, visualization and ML tools, a predictive model was made for the demand of each product and observed the need to use new models was identified such as Moving Averages and Exponential Smoothing to achieve a better forecast in posterior investigation.

The use of decision-making tools and planning parameters is very useful. An efficient forecasting model must best represent the behavior of sales and demand. The errors between the forecast and the actual demand should be as small as possible, this would be done with the use of different models with various forecasts and variables. With the aim of improving logistics and/or production processes, avoiding lost sales and excess inventory using expected demand, improving CX processes, identifying new trends, among others. Improvements to the ML prediction model must be made through exponential smoothing or moving averages, in order to better explain the behavior of sales for other types of products that linear regression does not sufficiently explain.
The application of this model using Machine Learning (ML) must be carried out with some test and training parameters; and with which the demand forecast for the following six months is made. In addition, since the products have different trends but most have an increasing or decreasing trend, it allows most of the data to be explained on a Linear regression model.
Once the model has carried out the tests and the training through the estimator yields the results, we can find thanks to the R^2 that this model does not explain all the products behavior sufficiently, but there are some with R^2 greater than 70%, which would explain in a good way the behavior of sales for those products.


One of the ways to make improvements to the prediction model is to add new variables (exogenous variables) that can explain in some way the number of units that are sold per product in each month.
