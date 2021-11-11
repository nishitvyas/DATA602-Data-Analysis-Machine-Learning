# Model Evaluation on Logistic Regression, Decision Tree and Support Vector Machine using Ensemble
The following project is concerned with the predictions on whether the flight will get delayed or not. There are total of 10K records and each record has 24 attributes. The project contains the visualization in the form of text and various graphs, the Python libraries which are used in this notebook are Numpy, Pandas, Matplotlib, Seaborn.
The modelling is done with the help of several modules of Scikit-Learn package.

# Dataset Description
 The dataset is a toy-dataset available on many sources such as [Kaggle](https://www.kaggle.com/aephidayatuloh/nyc-flights-2013/version/1?select=nyc_flights.csv) There are 24 columns(including target column) in total and about 10,000 records. This dataset linked in this project is merged data obtained from joining the flights, weather and planes tables. There are significant number of missing values in the dataset as either right or left join is used on planes to include every plane regardless of correspoding values are there or missing. These missing values are treated by the medians, most_frequent values for string and with the constants as per the need. 

# Business Question/Problem Statement
Predicting the delay of the flight based on the flight carrier details, weather details is the problem staement of this project. We will be tuning the model based on the accuracy. Based on the accuracy metric, we will be able to adjust the schedule of the flights with the respective of weather or other affecting factors.


# Exploratory Analysis Findings
- The 'arr_delay'(Target Class) is almost equally balanced, but there are about 264 records which does not have the target values so they are dropped.
- The values in the data weren't standardized so they are brought to the standard values in the modelling pipeline.
- The feature 'wind_dir' is not addressed in the model, there were significant missing values, but imputing the direction with the any central tendency can change the significance of that feature.
- Other features such as 'wind_gust' and 'precip' is kept zero for the missing values because the not available data for their attribute most probabaly signifies their absence so they are kept as zeros.


# Classification Results
We segemented our experiment into three parts:
- In the first part we have implemented the cross validation score on Logistic Regression(LogR), Decision Tree(DT) and Support Vector Machine(SVM). We got good accuracy on SVM but the deviation was of 2%, for DT and LogR it was 64% for both and the deviation of 2% and 1% repectively. 
- For the second part, we passed three models into an ensemble where the voting was selected as 'hard', since we are concerned about the accuracy, probability can be opted out.
- In the third part, we performed the GridSearch on the Ensemble to get the best estimator and its score, which was 67% (max of all)
- Lastly, we performed the AdaBoosting Classification where we kept 500 estimators but we got 64% which wasn't that effective as GridSearch.
- As conclusion, we should consider the Ensemble with the GridSearch as it gave maximum accuracy of all and also the deviation was 0%, it took more computations than other models.


# Potential Next Steps and Follow-ups
The consideration of the wind_dir, origin and dest needs to done for checking how the model predicts then.
Also the clarification on the weather of the origin or dest is not given.

