# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This project uses a simple Logistic Regression model for binary classification problem whether or not a person earns over or under $50k based on certain characteristics as dercribed in the dataset. The parameters used are 'max_iter' which is set to 1000 which indicates at which point the model should stop training if it fails to converge.

## Intended Use

This model is intended to use as a binary classifier, which can predict whether a person will earn over $50k or not, based on a avriety of factors as included int he dataset, such as - age, education, marital status, occupation, race, sex, etc. 

This is a simple model to be used for educational and coaching purposes, such as to teach students on basic ML and how to tune hyperparameters to get the highest accuracy. It can also be used to understand the concept of data slices and why model performance metrics are important across the slices, not just overall. 

It is not intended to be used in real-life cases to undertake a practical decision. 

## Training Data

The data is based on the census dataset from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/census+income).

To begin with, it was found that the column headers contains whitespaces in the beginning which made the eda.ipynb notebook to come up with errors. So the whitespaces were stripped. Then, based exploratory data analysis was performed in the notebook to understand the different columns and their values.

A One-Hot Encoder was used in the ML pipeline to encode the categorical features before splitting the data into 70% train set and 30% test set. The label, or the value to be predicted, is set as the column 'salary'. The final labels correpond to the salary brackets where 0 means < $50k and 1 means > $50k.

## Evaluation Data

For evaluation of the model, we use the remaining 30% of the dataset that we left out during training. This serves as our test set. Using this, we make the model make predictions and check those predictions agaisnt the actual values of the test set. Thsi works because the model has not seen the test set, thereby helping us evaluate the performance of our model using different metrics as detailed below. We use the same pre-processing steps for the evaluation data in our inference pipeline as we did for our training set.

## Metrics

To evaluate the model, we use metrics such as Precision, Recall and F-1 Score.

The overall metrics are:

1. Precision = 0.734
2. Recall = 0.563
3. F-1 Score = 0.637

## Ethical Considerations

When talkig about ethics, we need to keep in mind that machine learning models can create bias and unfairness due to the type of data is being fed. Because a machine cannot be held accountable, we should take all necessary steps we can in order to mitigate this bias. 

If we take a look at our current dataset, we will see that it uses features such as 'race', 'sex' and 'native-country'. Such features create bias and in a real-life scenario, it make risk making someone not being hired due to the model scoring less points for such individuals who hail from certain backgrounds just because they were not represeneted enough in the training set. And when certain proportions of backgrounds are not represented a lot in the dataset, the model learns that it is not as important or assigns less weights to it during training. This means that it will assign low score to it for example in an HR hiring system, but we as humans understand it is incorrect to discriminate in such a way.

Therefore for anyone using this dataset or creating another model for any other data, it is important to keep in mind that bias is not introduced in the training data.

## Caveats and Recommendations

Because the default hyperparameters are used, the precision is on the lower side and recall is also quiet low. Tuning these hyperparameters or experimenting with a different kind of classification model would lead to better results.

Trying to experiment in a way that model performance metrics are similar across all data slices, and not on a very different scale.

Another aspect would be to make the model fair and remove bias-creating features such as sex, race and native country.
