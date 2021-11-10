# Model Evaluation on Logistic Regression, Decision Tree and Support Vector Machine using Ensemble
The following project is concerned with the predictions on whether the flight will get delayed or not. There are total of 10K records and each record has 24 attributes. The project contains the visualization in the form of text and various graphs, the Python libraries which are used in this notebook are Numpy, Pandas, Matplotlib, Seaborn.
The modelling is done with the help of several modules of Scikit-Learn package.

# Dataset Description
 The dataset is a toy-dataset available on many sources such as [Kaggle](https://www.kaggle.com/aephidayatuloh/nyc-flights-2013/version/1?select=nyc_flights.csv) There are 24 columns(including target column) in total and about 10,000 records. This dataset linked in this project is merged data obtained from joining the flights, weather and planes tables. There are significant number of missing values in the dataset as either right or left join is used on planes to include every plane regardless of correspoding values are there or missing. These missing values are treated by the medians, most_frequent values for string and with the constants as per the need. 

# Business Question/Problem Statement
Predicting the delay of the flight based on the flight carrier details, weather details is the problem staement of this project. We will be tuning the model based on the accuracy. Based on the accuracy metric, we will be able to adjust the schedule of the flights with the respective of weather or other affecting factors.

The aim of this project is create a Logistic Regression/Classification model which will predict whether the person has Diabetes or not on the basis of 8 features. We will be tuning the model for obtaining true predictions. In this project we are focussing on the Recall metric, we will be able to capture all the actual positive Diabetic cases. The reason for this is because the more the accurate positive cases will be found earlier their treatments will be done. If the patient is actual Diabetes positive but predicted as negative, this case would be very expensive case, we want to  __minimize false negatives__ as much as we can.


# Exploratory Analysis Findings
- The __imbalance among the class__ is found as the number of positive cases are half of the number of negative cases, which is one of the drawback why we couldn't reach to a better accuracy.
- In the dataset, there were few records where the values in columns such as 'body mass index', 'serum insulin', 'diastolic blood pressure', 'triceps skinfold thickness' are zero which seems unreal. We have replaced these values keeping the relevancy of the values with respect to their classes(Positive/Negative). These unrealistic values caused a small distribution at extremeties, we have addressed the values which were zeros but for the values on the right extremity, they have kept the same keeping in mind that those can be real values.
- Overall, there were no missing values found in the data. 
- 
- 
- The 'arr_delay'(Target Class) is almost equally balanced, but there are about 264 records which does not have the target values so they are dropped.
- The values in the data weren't standardized so they are brought to the standard values in the modelling pipeline.
- The feature 'wind_dir' is not addressed in the model, there were significant missing values, but imputing the direction with the any central tendency can change the significance of that feature.
- Other features such as 'wind_gust' and 'precip' is kept zero for the missing values because the not available data for their attribute most probabaly signifies their absence so they are kept as zeros.
- 


# Classification Results
We segemented our experiment into three parts:
- In the first model where the data cleaning wasn't done, the AUROC(Area Under ROC) was 0.78 with the reacall of 0.47 for positive cases. The __False Negatives__ were __16__.
- In the second model, we ran Logistic Regression without Regularization and Principal Component Analysis (PCA), we got  AUROC of 0.835 which improved from the previous modelling. Also the __False Negatives__ were __10 from 16__.
- After cleaning and applying the Grid Search for tuning(Solvers: [Newton-CG, Saga, LBFGS], C=1, PCA=7), we got to know that the 7 features are most affecting out of 8 which tend to give a better recall value and also the AUROC, recall of 0.51 for positive and 0.88 for negative class was achieved along with the AUROC of 0.8374. Here the  number of __False Negative got increased to 12__ which seemed odd.
- So the better model from these was the __second one__ because we got minimum False Negatives. 


# Predictions Using this Model
From the classification report obtained, we got the confusion matrix '[[TP:85,FN:12][FP:28,TN:29]]' , there are still considerable False Negatives which should be classified as True Positives as per our aim to capture maximum positive cases. 

# Potential Next Steps and Follow-ups
We can try adding more positive class records in the data to create balance among the classes to reach to a better metric score.

