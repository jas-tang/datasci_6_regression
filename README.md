# datasci_6_regression
This is a repository for Assignment 7: Regressions in HHA507. 

## Simple Linear Regression

I first chose a [dataset](https://archive.ics.uci.edu/dataset/887/national+health+and+nutrition+health+survey+2013-2014+(nhanes)+age+prediction+subset) that has two numerical variables: a dependent variable and an independent variable. 
This was a National Health and Nutrition Health Survey 2013-2014 (NHANES) Age Prediction Subset. 

Imported these paackages
```
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from statsmodels.stats.diagnostic import linear_rainbow
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_goldfeldquandt
```

LBXIN Feature Continuous Respondent's Blood Insulin Levels -> Dependent Variable

LBXGLU Feature Continuous Respondent's Blood Glucose after fasting -> Independent Variable

Then I ran a predictor test. 
```
X = sm.add_constant(df['LBXIN'])  # Adds a constant term to the predictor
model = sm.OLS(df['LBXGLU'], X)
print(X)
```
With this as the result.
```
OLS Regression Results                            
==============================================================================
Dep. Variable:                 LBXGLU   R-squared:                       0.045
Model:                            OLS   Adj. R-squared:                  0.044
Method:                 Least Squares   F-statistic:                     107.0
Date:                Tue, 07 Nov 2023   Prob (F-statistic):           1.53e-24
Time:                        02:14:15   Log-Likelihood:                -9749.8
No. Observations:                2278   AIC:                         1.950e+04
Df Residuals:                    2276   BIC:                         1.952e+04
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const         94.9367      0.577    164.420      0.000      93.804      96.069
LBXIN          0.3901      0.038     10.345      0.000       0.316       0.464
==============================================================================
Omnibus:                     3088.205   Durbin-Watson:                   2.090
Prob(Omnibus):                  0.000   Jarque-Bera (JB):           924152.102
Skew:                           7.497   Prob(JB):                         0.00
Kurtosis:                     100.528   Cond. No.                         24.2
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
```
The R-squared value of 0.045, meaning an approximate 4%, tells us that our dependent variable probably does not have a strong strong correlation for predicting the independent variable. The Respondent's Blood Insulin Levels does not really predict the Respondent's Blood Glucose after fasting. 4% of the variance that we see in Blood glucose level after fasting can be attributed to the blood insulin levels.

To assess the linearity, we ran a couple of more code lines.
```
residuals = results.resid
fitted = results.fittedvalues
```
```
### Assessing linearity of the relationship
stat, p_value = linear_rainbow(results)
print(f"Rainbow Test: stat={stat}, p-value={p_value}")
```
Our p-value ended up being p-value=0.9999999999999999. Interestingly, the p-value seems to be greater than .05, meaning that there is a linear relationship.

Graphing the linear regression.
![](https://github.com/jas-tang/datasci_6_regression/blob/main/images/1.JPG)

We then assessed the normality of the residuals. 
```
### Assessing normality of the residuals
W, p_value = shapiro(residuals)
print(f"Shapiro-Wilk Test: W={W}, p-value={p_value}")
```
The p-value was p-value=0.0, meaning there is not a normal distribution of residuals.

Plotting the residuals
![](https://github.com/jas-tang/datasci_6_regression/blob/main/images/2.JPG)

Then we assessed the homegeneity of variance of the residuals. 
```
##### Assessing the homogeneity of variance of the residuals
gq_test = het_goldfeldquandt(residuals, results.model.exog)
print(f"Goldfeld-Quandt Test: F-statistic={gq_test[0]}, p-value={gq_test[1]}")
```
Goldfeld-Quandt Test: F-statistic=1.1971619635468016, p-value=0.001217680059819782

This P-value is lower than .05, meaning that the homogeneity of variance has not been met.

To summarize, our R-squared tells us that that The Respondent's Blood Insulin Levels does not really predict the Respondent's Blood Glucose after fasting. The relationship is said to be linear. The normality is not normally distributed. The homogeneity hass also not been met.

## Multiple Linear Regression
We chose a different [dataset](https://archive.ics.uci.edu/dataset/110/yeast) that contained at least five continuous variables. 
This was a dataset that focused on predicting yeast. 

Dependent Variable: 
* alm: Score of the ALOM membrane spanning region prediction program. 

Independent Variables: 
* mit: Score of discriminant analysis of the amino acid content of the N-terminal region (20 residues long) of mitochondrial and non-mitochondrial proteins.
* erl: Presence of HDEL substring (thought to act as a signal for retention in the endoplasmic reticulum lumen). Binary attribute.
* pox: Peroxisomal targeting signal in the C-terminus.
* vac: Score of discriminant analysis of the amino acid content of vacuolar and extracellular proteins.
* nuc: Score of discriminant analysis of nuclear localization signals of nuclear and non-nuclear proteins.

We created a separate variable that contained our independent variables. 
```
listofpredictors = df2[['mit', 'erl', 'pox', 'vac', 'nuc']]
```
Then we ran predictability.
```
# Fit the regression model
X = sm.add_constant(listofpredictors)  # Adds a constant term to the predictor
y = df2['alm']
model = sm.OLS(y, X)
results = model.fit()
```
With this as the result.
```
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                    alm   R-squared:                       0.036
Model:                            OLS   Adj. R-squared:                  0.033
Method:                 Least Squares   F-statistic:                     11.16
Date:                Tue, 07 Nov 2023   Prob (F-statistic):           1.43e-10
Time:                        03:44:48   Log-Likelihood:                 1551.6
No. Observations:                1484   AIC:                            -3091.
Df Residuals:                    1478   BIC:                            -3059.
Df Model:                           5                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const          0.6298      0.030     20.693      0.000       0.570       0.689
mit            0.0258      0.016      1.588      0.112      -0.006       0.058
erl            0.0004      0.046      0.009      0.993      -0.089       0.090
pox            0.0154      0.029      0.526      0.599      -0.042       0.073
vac           -0.2723      0.039     -7.040      0.000      -0.348      -0.196
nuc           -0.0025      0.021     -0.118      0.906      -0.043       0.039
==============================================================================
Omnibus:                       59.461   Durbin-Watson:                   1.480
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               84.346
Skew:                          -0.380   Prob(JB):                     4.84e-19
Kurtosis:                       3.887   Cond. No.                         30.6
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
```
Our adjusted R-squared is .03, or 3%. This means that our independent variables has little predictability for our dependent variable.

![](https://github.com/jas-tang/datasci_6_regression/blob/main/images/3.JPG)

Based off the p- values, mit, erl, pox, and nuc are all not significant while vac is. We can theoretically drop the not significant values for a backwards regression.

```
listofpredictors2 = df2[['vac']]

# Fit the regression model
X = sm.add_constant(listofpredictors2)  # Adds a constant term to the predictor
y = df2['alm']
model = sm.OLS(y, X)
results = model.fit()
```
This was the result.
```
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                    alm   R-squared:                       0.035
Model:                            OLS   Adj. R-squared:                  0.034
Method:                 Least Squares   F-statistic:                     52.99
Date:                Tue, 07 Nov 2023   Prob (F-statistic):           5.41e-13
Time:                        03:42:24   Log-Likelihood:                 1550.2
No. Observations:                1484   AIC:                            -3096.
Df Residuals:                    1482   BIC:                            -3086.
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const          0.6393      0.019     33.193      0.000       0.602       0.677
vac           -0.2786      0.038     -7.280      0.000      -0.354      -0.204
==============================================================================
Omnibus:                       61.851   Durbin-Watson:                   1.475
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               92.460
Skew:                          -0.375   Prob(JB):                     8.37e-21
Kurtosis:                       3.966   Cond. No.                         21.6
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
```
We see a very slight increase in R-Squared.

Going back to using multiple independent variables because we had to remove 4 of them after the backwards regression. 

```
### Assessing linearity of the relationship
stat, p_value = linear_rainbow(results)
print(f"Rainbow Test: stat={stat}, p-value={p_value}")
```
Rainbow Test: stat=1.1546391594034604, p-value=0.025433623259621063

A significant p-value indicates that the relationship is not linear.

```
residuals = results.resid
fitted = results.fittedvalues

### Assessing normality of the residuals
W, p_value = shapiro(residuals)
print(f"Shapiro-Wilk Test: W={W}, p-value={p_value}")
```
Shapiro-Wilk Test: W=0.973564624786377, p-value=7.147997221826753e-16
There is a normal distribution of residuals because p-value is less than .05.

![](https://github.com/jas-tang/datasci_6_regression/blob/main/images/4.JPG)

```
##### Assessing the homogeneity of variance of the residuals
gq_test = het_goldfeldquandt(residuals, results.model.exog)
print(f"Goldfeld-Quandt Test: F-statistic={gq_test[0]}, p-value={gq_test[1]}")
```
Goldfeld-Quandt Test: F-statistic=1.1340584877887678, p-value=0.044066389293097995

This P-value is lower than .05, meaning that the homogeneity of variance has not been met.

We plot the residuals vs fitted values.
![](https://github.com/jas-tang/datasci_6_regression/blob/main/images/5.JPG)

Now we have to check the multicolinearity using VIF. 
```
from statsmodels.stats.outliers_influence import variance_inflation_factor
```

```
# Checking multicollinearity using VIF
vif_data = pd.DataFrame()
vif_data['Variable'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print("\nVIF Data:")
print(vif_data)
```
This was the result: 
```

VIF Data:
  Variable         VIF
0    const  189.255945
1      mit    1.013064
2      erl    1.002024
3      pox    1.002049
4      vac    1.020608
5      nuc    1.011709
```
There are no independent variables that are highly correlated with any of the other independent variables that exist in our model as all of the VIF values are less than 10.
