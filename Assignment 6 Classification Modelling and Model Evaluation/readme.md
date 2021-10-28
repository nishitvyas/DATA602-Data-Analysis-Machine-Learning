# Model Evaluation on Logistic Regression
This project is about the classification of the records having Diabetes or not. The project is an example of Supervised learning as we have the target_values. On the basis of the 8 features, it is decided whether the person has diabetes. The Logistic Regression model is used for this dataset.
The project contains the visualization in the form of text and various graphs, the Python libraries which are used in this notebook are Numpy, Pandas, Matplotlib, Seaborn.
The modelling is done with the help of several modules of Scikit-Learn package.

# Dataset Description
 The dataset is a toy-dataset available on many sources such as [Kaggle](https://www.kaggle.com/uciml/pima-indians-diabetes-database), [Network Repository](https://networkrepository.com/pima-indians-diabetes.php#:~:text=Metadata%20%20%20Name%20%20%20Pima%20Indians,%20%20768%20%204%20more%20rows%20) There are 9 columns(including target column) in total and about 768 records. There are no missing values in the dataset. There is some unrealistic values on columns such as 'Body Mass Index' which is cleaned in the notebook for enhacing the accuracy of the model.

# Business Question/Problem Statement
The aim of this project is create a Logistic Regression/Classification model which will predict whether the person has Diabetes or not on the basis of 8 features. We will be tuning the model for obtaining true predictions. In this project we are focussing on the Recall metric, we will be able to capture all the actual positive Diabetic cases. The reason for this is because the more the accurate positive cases will be found earlier their treatments will be done. If the patient is actual Diabetes positive but predicted as negative, this case would be very expensive case, we want to  __minimize false negatives__ as much as we can.


# Exploratory Analysis Findings
- The __imbalance among the class__ is found as the number of positive cases are half of the number of negative cases, which is one of the drawback why we couldn't reach to a better accuracy.
- In the dataset, there were few records where the values in columns such as 'body mass index', 'serum insulin', 'diastolic blood pressure', 'triceps skinfold thickness' are zero which seems unreal. We have replaced these values keeping the relevancy of the values with respect to their classes(Positive/Negative). These unrealistic values caused a small distribution at extremeties, we have addressed the values which were zeros but for the values on the right extremity, they have kept the same keeping in mind that those can be real values.
- Overall, there were no missing values found in the data. 
- The values in the data weren't standardized so they are brought to the standard values in the modelling pipeline.


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
