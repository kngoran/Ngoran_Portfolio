# KNgoran Portfolio
## My Data Science Portfolio


## [Project 1 : Predict london boroughs House Price Using RegressionAlgorithms] (https://github.com/kngoran/Project1/blob/main/London%20Boroughs%20House%20Price%20Prediction%20.ipynb)


### Objectives


In this notebook, we are  going to apply our data science skills to solve a real life problem.

Here’s the mystery we are going to solve: which boroughs of London have seen the greatest increase in housing prices, on average, over the last two decades?

A borough is just a fancy word for district. You may be familiar with the five boroughs of New York… well, there are 32 boroughs within Greater London (here's some info for the curious). Some of them are more desirable areas to live in, and the data will reflect that with a greater rise in housing prices


Data Science is magical. In this case study, we will get to apply some complex machine learning algorithms. But as David Spiegelhalter reminds us, there is no substitute for simply taking a really, really good look at the data. Sometimes, this is all we need to answer our question.

Data Science projects generally adhere to the four stages of Data Science Pipeline:

##### Sourcing and loading
##### Cleaning, transforming, and visualizing
##### Modeling
##### Evaluating and concluding

For this project we will use:
Python essentially pandas matplotlib for
    data ingestion and inspection 
    Exploration data analysis
    Tidying  and cleaning the  dataset
    Transforming and  Subsetting the  DataFrames with list 
    Filtering and grouping the
    Melting the data 
    Handling date and date-time 
    
 





## [Project 2: Decision Tree Specialty Coffee Case Study](https://github.com/kngoran/Project1/blob/main/%20Decision%20Tree%20Specialty%20Coffee%20Case%20Study%20Right%20Copy%20.ipynb)


### Objectives 

As a data scientist, I  will build a decision tree to predict how many units of the Hidden Farm Chinese coffee will be purchased by RR Diner Coffee's most loyal customers.
To this end, we have a survey of 710 of the most loyal RR Diner Coffee customers, collecting data on the customers':
    age
    gender
    salary
    whether they have bought at least one RR Diner Coffee product online
    their distance from the flagship store in the USA (standardized to a number between 0 and 11)
    how much they spent on RR Diner Coffee products on the week of the survey
    how much they spent on RR Diner Coffee products in the month preceding the survey
    the number of RR Diner coffee bean shipments each customer has ordered over the preceding year

 Also , I asked each customer participating in the survey whether they would buy the Hidden Farm coffee, and some (but not all) of the customers gave responses to that question.

 If more than 70% of the interviewed customers are likely to buy the Hidden Farm coffee, you will strike the deal with the local Hidden Farm farmers and sell the coffee. Otherwise, we won't strike the deal and the Hidden Farm coffee will remain in legends only. There's some doubt in your mind about whether 70% is a reasonable threshold, but it'll do for the moment.

To solve the problem, then, you will build a decision tree to implement a classification solution.

As always, We will follow the four stages of Data Science Pipeline:

1. Sourcing and loading

    Import packages,
    Load data,
    Explore the data.

2. Cleaning, transforming and visualizing

    Cleaning the data,
    Train/test split.

3. Modeling

    Model 1: Entropy model - no max_depth,
    Model 2: Gini impurity model - no max_depth,
    Model 3: Entropy model - max depth 3,
    Model 4: Gini impurity model - max depth 3.

4. Evaluating and concluding

    How many customers will buy Hidden Farm coffee?
    Decision,

5. Random Forest

    Import necessary modules,
    Model,
    Revise conclusion.



## [Project 3: Regression: The Red Wine Dataset](https://github.com/kngoran/SpringBoardProjects/blob/master/%20Regression%20Case%20Study-the%20Red%20Wine%20Dataset.ipynb) 

This case study was designed  to use Python to apply the knowledge you've acquired in reading The Art of Statistics (hereinafter AoS) by Professor Spiegelhalter. Specifically,  we will a regression analysis; a method discussed in Chapter 5 on p.121. It might be useful to have the book open at that page when doing the case study to remind you of what it is we're up to (but bear in mind that other statistical concepts, such as training and testing, will be applied, so you might have to glance at other chapters too).

## [Project 4 Predict Employees Attrition and Retention Using Classification algorithms](https://github.com/kngoran/SpringBoardProjects/blob/master/Capstone%20Project%202%20Employees'%20Turnover%20%26%20Retention2.ipynb)

Turnover is the number of employees that leave , voluntarily or involuntarily, an organization in a period of time. In either way, companies have to replace those employees to keep the optimal operation of the organization. The extent to which employers face turnover rates and the cost often varies by organizations and industries. Depending on industries, management has to pay particular attention to the turnover rate since it can reduce the productivity and loss of clients that affect the company's long-run bottom line. In this project, we will try to uncover the key variables that contribute to employees' turnover.


## [Project 5: Customer Segmentation with K-Means"](https://github.com/kngoran/SpringBoardProjects/blob/master/Clustering%20Case%20Study%20-%20Customer%20Segmentation%20with%20K-Means.ipynb)


This case study is based on this blog post by the yhat blog. Please feel free to refer to the post for additional information, and solutions.

Structure of the mini-project:

Sourcing and loading
Load the data
Explore the data
Cleaning, transforming and visualizing
Data Wrangling: Exercise Set 1
Creating a matrix with a binary indicator for whether they responded to a given offer
Ensure that in doing so, NAN values are dealt with appropriately
Modelling

K-Means clustering: Exercise Sets 2 and 3

Choosing K: The Elbow method
Choosing K: The Silhouette method
Choosing K: The Gap statistic method
Visualizing clusters with PCA: Exercise Sets 4 and 5

Conclusions and next steps
Conclusions
Other clustering algorithms (Exercise Set 6)

## [Project 6: Gradient Boosting](https://github.com/kngoran/SpringBoardProjects/blob/master/Gradient%20Boosting%20Case_Study.ipynb)

 The gradient descent algorithm in the context of fitting linear regression models. For a particular regression model with n parameters, an n+1 dimensional space existed defined by all the parameters plus the cost/loss function to minimize. The combination of parameters and loss function define a surface within the space. The regression model is fitted by moving down the steepest 'downhill' gradient until we reach the lowest point of the surface, where all possible gradients are 'uphill.' The final model is made up of the parameter estimates that define that location on the surface.

Throughout all iterations of the gradient descent algorithm for linear regression, one thing remains constant: The underlying data used to estimate the parameters and calculate the loss function never changes. In gradient boosting, however, the underlying data do change.

Each time we run a decision tree, we extract the residuals. Then we run a new decision tree, using those residuals as the outcome to be predicted. After reaching a stopping point, we add together the predicted values from all of the decision trees to create the final gradient boosted prediction.

Gradient boosting can work on any combination of loss function and model type, as long as we can calculate the derivatives of the loss function with respect to the model parameters. Most often, however, gradient boosting uses decision trees, and minimizes either the residual (regression trees) or the negative log-likelihood (classification trees).

Let’s go through a simple regression example using Decision Trees as the base predictors (of course Gradient Boosting also works great with regression tasks). This is called Gradient Tree Boosting, or Gradient Boosted Regression Trees. First, let’s fit a DecisionTreeRegressor to the training set.

