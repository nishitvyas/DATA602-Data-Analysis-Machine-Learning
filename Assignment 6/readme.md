# Model Evaluation on Logistic Regression {Work in Progress}
This project is about the classification of the records having Diabetes or not. The project is an example of Supervised learning as we have the target_values. On the basis of the 8 features, it is decided whether the person has diabetes. The Logistic Regression model is used for this dataset.
The project contains the visualization in the form of text and various graphs, the Python libraries which are used in this notebook are Numpy, Pandas, Matplotlib, Seaborn.
The modelling is done with the help of several modules of Scikit-Learn package.

# Dataset Description
 The dataset is a toy-dataset available on many sources such as [Kaggle](https://www.kaggle.com/uciml/pima-indians-diabetes-database), [Network Repository](https://networkrepository.com/pima-indians-diabetes.php#:~:text=Metadata%20%20%20Name%20%20%20Pima%20Indians,%20%20768%20%204%20more%20rows%20) There are 9 columns(including target column) in total and about 768 records. There are no missing values in the dataset. There is some unrealistic values on columns such as 'Body Mass Index' which is cleaned in the notebook for enhacing the accuracy of the model.

# Business Question/Problem Statement
The aim of this project is create a Logistic Regression/Classification model which will predict whether the person has Diabetes or not on the basis of 8 features. We will be tuning the model for obtaining true predictions. In this project we are focussing on the Recall metric, we will be able to capture all the actual positive Diabetic cases. The reason for this is because the more the accurate positive cases will be found earlier their treatments will be done. 



# Exploratory Analysis Findings
The imbalance among the class is found as the number of positive cases are half of the number of negative cases, which is one of the drawback why we couldn't reach to a better accuracy.
In the dataset, there were few records where the values in columns such as 'body mass index', 'serum insulin', 'diastolic blood pressure', 'triceps skinfold thickness' are zero which seems unreal. We have replaced these values keeping the relevancy of the values with respect to their classes(Positive/Negative). These unrealistic values caused a small distribution at extremeties, we have addressed the values which were zeros but for the values on the right extremity, they hve kept the same keeping in mind that those can be real values.
Overall, there were no missing values found in the data. 
The values in the data weren't standardized so they are brought to the standard values in the modelling pipeline.



# Regression Results
We got the R-squared as 0.76 which means we were able to capture the 76% variance of data. We came to know that the city in which the house is constructed is one of the major driver or we can say the area in which the house is, plays an important role. Other factors such as area-house, number of bedrooms, resale, rera are also the driving factors. We observed that the area column was creating multi-collinearity, removing it from the features helped increasing the R-squared by 10%.
# Predictions Using this Dataset
We predicted the house prices using Linear Regression with the score of 45% in the training data and 47% on test data.

# Potential Next Steps and Follow-ups
We want to categorised the cities into the 'tier level' so that the categorisation becomes easy and check how the model works.
