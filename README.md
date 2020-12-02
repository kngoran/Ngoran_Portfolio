# Ngoran_Portfolio
## My Data Science Portfolio


## [Project 1 : Predict london_borough House Price](https://github.com/kngoran/Project-/blob/main/%20%20Tier3%20London_Boroughs%20_project.ipynb)


Objectives
Hey! Great job getting through those challenging DataCamp courses. You're learning a lot in a short span of time.

In this notebook, you're going to apply the skills you've been learning, bridging the gap between the controlled environment of DataCamp and the slightly messier work that data scientists do with actual datasets!

Here’s the mystery we’re going to solve: which boroughs of London have seen the greatest increase in housing prices, on average, over the last two decades?

A borough is just a fancy word for district. You may be familiar with the five boroughs of New York… well, there are 32 boroughs within Greater London (here's some info for the curious). Some of them are more desirable areas to live in, and the data will reflect that with a greater rise in housing prices.

This is the Tier 3 notebook, which means it's not filled in at all: we'll just give you the skeleton of a project, the brief and the data. It's up to you to play around with it and see what you can find out! Good luck! If you struggle, feel free to look at easier tiers for help; but try to dip in and out of them, as the more independent work you do, the better it is for your learning!

This challenge will make use of only what you learned in the following DataCamp courses:

Prework courses (Introduction to Python for Data Science, Intermediate Python for Data Science)
Data Types for Data Science
Python Data Science Toolbox (Part One)
pandas Foundations
Manipulating DataFrames with pandas
Merging DataFrames with pandas
Of the tools, techniques and concepts in the above DataCamp courses, this challenge should require the application of the following:

pandas
data ingestion and inspection (pandas Foundations, Module One)
exploratory data analysis (pandas Foundations, Module Two)
tidying and cleaning (Manipulating DataFrames with pandas, Module Three)
transforming DataFrames (Manipulating DataFrames with pandas, Module One)
subsetting DataFrames with lists (Manipulating DataFrames with pandas, Module One)
filtering DataFrames (Manipulating DataFrames with pandas, Module One)
grouping data (Manipulating DataFrames with pandas, Module Four)
melting data (Manipulating DataFrames with pandas, Module Three)
advanced indexing (Manipulating DataFrames with pandas, Module Four)
matplotlib (Intermediate Python for Data Science, Module One)
fundamental data types (Data Types for Data Science, Module One)
dictionaries (Intermediate Python for Data Science, Module Two)
handling dates and times (Data Types for Data Science, Module Four)
function definition (Python Data Science Toolbox - Part One, Module One)
default arguments, variable length, and scope (Python Data Science Toolbox - Part One, Module Two)
lambda functions and error handling (Python Data Science Toolbox - Part One, Module Four)
The Data Science Pipeline
This is Tier Three, so we'll get you started. But after that, it's all in your hands! When you feel done with your investigations, look back over what you've accomplished, and prepare a quick presentation of your findings for the next mentor meeting.

Data Science is magical. In this case study, you'll get to apply some complex machine learning algorithms. But as David Spiegelhalter reminds us, there is no substitute for simply taking a really, really good look at the data. Sometimes, this is all we need to answer our question.

Data Science projects generally adhere to the four stages of Data Science Pipeline:

Sourcing and loading
Cleaning, transforming, and visualizing
Modeling
Evaluating and concluding

## [Project 2: Decision Tree Specialty Coffee Case Study](https://github.com/kngoran/SpringBoardProjects/blob/master/%20Decision%20Tree%20Specialty%20Coffee%20Case%20Study.ipynb)

Imagine you've just finished the Springboard Data Science Career Track course, and have been hired by a rising popular specialty coffee company - RR Diner Coffee - as a data scientist. Congratulations!

RR Diner Coffee sells two types of thing:

specialty coffee beans, in bulk (by the kilogram only)
coffee equipment and merchandise (grinders, brewing equipment, mugs, books, t-shirts).
RR Diner Coffee has three stores, two in Europe and one in the USA. The flagshap store is in the USA, and everything is quality assessed there, before being shipped out. Customers further away from the USA flagship store have higher shipping charges.


## [Project 3: Regression: The Red Wine Dataset](https://github.com/kngoran/SpringBoardProjects/blob/master/%20Regression%20Case%20Study-the%20Red%20Wine%20Dataset.ipynb) 

Welcome to the Unit 8 Springboard Regression case study! Please note: this is Tier 3 of the case study.

This case study was designed for you to use Python to apply the knowledge you've acquired in reading The Art of Statistics (hereinafter AoS) by Professor Spiegelhalter. Specifically, the case study will get you doing regression analysis; a method discussed in Chapter 5 on p.121. It might be useful to have the book open at that page when doing the case study to remind you of what it is we're up to (but bear in mind that other statistical concepts, such as training and testing, will be applied, so you might have to glance at other chapters too).

## [Project 3 Predict Employees Attrition and Retention Using Classification algorithms](https://github.com/kngoran/SpringBoardProjects/blob/master/Capstone%20Project%202%20Employees'%20Turnover%20%26%20Retention2.ipynb)

Many organizations have found that turnover is a very costly problem. Turnover is a process in which employees leave an organization. Employees leave the company voluntarily or involuntarily. In either way, companies have to replace them. The extent to which employers face turnover rates and the cost often varies by organizations and industries. Depending on the organization or industries, management has to pay particular attention to the turnover rate since it can reduce the productivity and loss of clients that affect the company's long-run bottom line. In this project, we will try to uncover the key variables that contribute to employees' turnover.


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

You may recall that we last encountered gradients when discussing the gradient descent algorithm in the context of fitting linear regression models. For a particular regression model with n parameters, an n+1 dimensional space existed defined by all the parameters plus the cost/loss function to minimize. The combination of parameters and loss function define a surface within the space. The regression model is fitted by moving down the steepest 'downhill' gradient until we reach the lowest point of the surface, where all possible gradients are 'uphill.' The final model is made up of the parameter estimates that define that location on the surface.

Throughout all iterations of the gradient descent algorithm for linear regression, one thing remains constant: The underlying data used to estimate the parameters and calculate the loss function never changes. In gradient boosting, however, the underlying data do change.

Each time we run a decision tree, we extract the residuals. Then we run a new decision tree, using those residuals as the outcome to be predicted. After reaching a stopping point, we add together the predicted values from all of the decision trees to create the final gradient boosted prediction.

Gradient boosting can work on any combination of loss function and model type, as long as we can calculate the derivatives of the loss function with respect to the model parameters. Most often, however, gradient boosting uses decision trees, and minimizes either the residual (regression trees) or the negative log-likelihood (classification trees).

Let’s go through a simple regression example using Decision Trees as the base predictors (of course Gradient Boosting also works great with regression tasks). This is called Gradient Tree Boosting, or Gradient Boosted Regression Trees. First, let’s fit a DecisionTreeRegressor to the training set.

