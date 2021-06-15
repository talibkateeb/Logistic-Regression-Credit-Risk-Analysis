# Logistic-Regression-Credit-Risk-Analysis

## Analysis Report

### Overview:

The goal of this porject is to perform a logisitc regression classification on the credit loans dataset, in order to predict healthy loans vs. high-risk loans, and evaluate the accuracy of that prediction, using both the original dataset and an oversampled dataset. 

The data consisted of the loan size, interest rate, borrower income, debt to income ratio, number of accounts, derogaroty marks, total debt, and loan status. I needed to predict the loan status '0' (healthy loan) or '1' (high-risk loan). For the original sample of data the value counts for the target prediction was 75036 healthy loans vs. 2500 high-risk loans, which shows imbalanced classes in the model. 

I started by reading the CSV data into a pandas dataframe. Then I split the data into X (features) and y (target) dataframes. I checked the value counts to verify the imbalance. I then split the data into training and testing data to be used with the logistic regression model. I then initialized the model, fit it onto the training data, and used the fitted model to predict the testing data. I then assessed the model's performance by evaluating the balanced accuracy score, the confusion matrix, and the classification report. I the performed random oversampling to the dataset to fix the imbalance, then performed the same steps of model-fit-predict-evaluate, this time with the oversampled data fit on the logistic regression model. 

### Results

Machine Learning Model 1 (original dataset):

* Balanced Accuracy Score: ~0.952 or ~95% 

* Precision Score: 0.99 - (1.00 for healthy loans, 0.85 for high-risk loans)

* Recall Score: 0.99 - (0.99 for healthy loans, 0.91 for high-risk loans)


Machine Learning Model 2 (randomly oversampled dataset):

* Balanced Accuracy Score: ~0.99367 or ~99.4% 

* Precision Score: 0.99 - (1.00 for healthy loans, 0.84 for high-risk loans)

* Recall Score: 0.99 - (0.99 for healthy loans, 0.99 for high-risk loans)

### Summary 

The first model appears to predict both of them with really well. It predicts the healthy loan almost perfectly, and predicts the high risk loan a little less accuratley but still very high. Both their precision and recall scores are high as well as their F-1 score. The healthy loan has perfect on 2/3 and 0.99 on the recall. While the high risk loan has a 0.85 precision, 0.91 recall, and 0.88 F-1 score. However, due to the imbalance we cannot be sure that this is actually true, and that the results are not skewed due to the low value counts of the high risk loans. Overall, the logistic regression model fit with oversampled data predicts the healthy and high risk loans better than the original non-oversampled data. Even though the original model had high scores for accuracy, precision, recall, and F-1, it appears the new oversampled model had higher scores in all categories. So the oversampled model predicts better than the one fit with original data. And since I had randomly oversampled the data, we can be sure that the imbalance does not affect the results. 

I recommend using the oversampled model since it has a higher recall score for the high-risk loans. Based on our problem it is more important to correctly classify a high-risk loan correctly than a healthy loan correctly. Since the randomly oversampled data has more instanced to learn from, it performs better at predicting the high-risk loans. 

---

## Technologies

This application runs on python version 3.7, with the following add-ons:

* [Jupyter Lab/Notebook](https://jupyter.org/) - A development tool for interactive programming.

* [Pandas](https://pandas.pydata.org/) - A python librart for data analysis and manipulation.

* [scikit-learn](https://scikit-learn.org/) - A python library for Machine Learning tools.

* [Numpy](https://numpy.org/) - a python library for scientific and mathematical computing. 

* [imbalanced-learn](https://imbalanced-learn.org/) - A python library that relies on scikit-learn and provides tools for imbalanced classes in classification. 

---

## Installation Guide

Download and install [Anaconda](https://www.anaconda.com/products/individual-b)

Open terminal and install Pandas by running this command:

    pip install pandas

    pip install -U scikit-learn

Activate Conda environment:

    conda activate "environment name"

    conda install -c conda-forge imbalanced-learn


Open Jupyter Notebook and navigate to the notebook file. 

Click on each cell to run individually:

    Shift + Enter

---

## Example

Running these 2 cells shows the confusion matrix and the classification report of the logistic regression prediction from the model.

![Code Example]()

---

## Contributors

*  Talib Kateeb

---

## License

[Click Here To View](https://github.com/talibkateeb/Logistic-Regression-Credit-Risk-Analysis/blob/main/LICENSE)
