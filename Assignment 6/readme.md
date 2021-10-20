# Model Evaluation on Logistic Regression {Work in Progress}
This project is about the housing price in the various cities of India. We want to explore the data and find how the house prie is calculated. 

# Dataset Description
 The dataset is taken from the [Kaggle](https://www.kaggle.com/ishandutta/machine-hack-housing-price-prediction) There are 12 columns in total and about 29451 records. There are     no missing values in the dataset. 

# Business Question/Problem Statement
The purpose of this project is to know the pattern on how the features(columns) are affecting the house price. The project contains the visualization in the form of text and various graphs, the Python libraries which are used in this notebook are Numpy, Pandas, Matplotlib, Seaborn. The first part of this project is Exploratory Data Analysis and the second part will be about the regression model, predicting values and model evaluation.


# Exploratory Analysis Findings
The data has about 1500 outlier values which can cause inaccuracy in our model. The range of price is (0.25-3000) where the median is 60, this states that 50% of the records is under 60. About 5-10% of data is deviating(having the price value very high with respect to the mdeian)
As conclusion, it is clear that the price of house depends on the location of the property, area of the property, age of property/('RESALE') and number of rooms in that property. Although the construction year wasn't mentioned we couldn't derive the age correctly but taken an assumption that the property which is in the 'RESALE' maybe considerably old.


# Regression Results
We got the R-squared as 0.76 which means we were able to capture the 76% variance of data. We came to know that the city in which the house is constructed is one of the major driver or we can say the area in which the house is, plays an important role. Other factors such as area-house, number of bedrooms, resale, rera are also the driving factors. We observed that the area column was creating multi-collinearity, removing it from the features helped increasing the R-squared by 10%.
# Predictions Using this Dataset
We predicted the house prices using Linear Regression with the score of 45% in the training data and 47% on test data.

# Potential Next Steps and Follow-ups
We want to categorised the cities into the 'tier level' so that the categorisation becomes easy and check how the model works.
