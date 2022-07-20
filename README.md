# DATA310-Project-2

This project was compleated as a part of DATA 310 at William & Mary. The project consisted of one programming assignments. In addition to the README file, this repository contains a folder which has the .ipynb files for all questions.

## Question One

Question One Asks the following:

"Import (including the correct names of the variables) the data from the QSAR Fish Toxicity research: https://archive.ics.uci.edu/ml/datasets/QSAR+fish+toxicity On this webpage you will find a description of the explanatory variables and the dependent variable. Implement 10-fold cross validations and the correct data preprocessing, make polynomial models of degree 1, 2 and 3 and determine which degree achieves the best external validation.  For this, you will use the regularized regression models such as Ridge, Lasso and ElasticNet and you will also determine the best value of the hyperparameter alpha in each case. In the case of ElasticNet, the second hyperparameter, L1 ratio, can be assumed to be 0.5. 

After you obtained the best choice of polynomial degree and hyperparameter alpha compute the residuals of the best model on the whole data and determine whether they follow a normal distribution."

This project began by importing the QSAR Fish Toxicity research from the University of California - Irvines Machine Learning Repository. Since I used Google Colab for this assignment I imported the data using the following code.

```Python
dataframe = pd.read_csv('drive/MyDrive/DATA 310/Project 2/qsar_fish_toxicity.csv', delimiter = ';' ,names = ['CIC0','SM1_Dz(Z)','GATS1i','NdsCH','NdssC','MLOGP','LC50 [-LOG(mol/L)]'])
X = dataframe.iloc[:,:-1].values
y = dataframe.iloc[:,-1].values
```
It is worth noting that the data set itself features unnamed columns, however the columns are identified on the Machine Learning Repository's website. Columns 1-6 represent the features, while Column 7 is the target.

The next step I took was to write a function that could be used to test and identify any combination of polynomial features,linear models, and alpha parameters. I did so using a series of "for loops" and an internal K-Fold cross validation. The code that I produced is below.

```Python
def Project2 (model, X, y, k, degrees, a_start, a_end, a_tests, random_state=123):
  import numpy as np
  from sklearn.model_selection import KFold
  from sklearn.preprocessing import StandardScaler, PolynomialFeatures
  from sklearn.linear_model import Ridge, Lasso, ElasticNet
  from sklearn.pipeline import Pipeline
  import matplotlib.pyplot as plt
  from sklearn.svm import SVR

  scale = StandardScaler()

  R2train_avg = []
  R2test_avg = []
  degree_value =[]
  a_value = []

  kf = KFold(n_splits=k, random_state=random_state, shuffle=True)

  for d in range (1,degrees+1,1):
    poly = PolynomialFeatures(degree = d)
    pipe = Pipeline([['Scaler',scale],['Poly Feats',poly]])

    for a in np.linspace(a_start,a_end,a_tests):
      test_model = model(alpha= a, fit_intercept = False, max_iter=5000)

      R2train_raw = []
      R2test_raw = []

      for idxtrain, idxtest in kf.split(X):
        Xtrain = X[idxtrain]
        Xtest = X[idxtest]
        ytrain = y[idxtrain]
        ytest = y[idxtest]
        Xpolytrain = pipe.fit_transform(Xtrain)
        Xpolytest = pipe.transform(Xtest)

        #Regression
        test_model.fit(Xpolytrain,ytrain)
        R2train_raw.append(test_model.score(Xpolytrain,ytrain))
        R2test_raw.append(test_model.score(Xpolytest,ytest))
      R2test_avg.append(np.mean(R2test_raw))
      R2train_avg.append(np.mean(R2train_raw))
      degree_value.append(d)
      a_value.append(a)
  
  return R2test_avg, degree_value, a_value
  ```
  The user of this function determines the following imports:
  - The desired model
  - The X data
  - The y data
  - The number of splits in the K-Fold
  - The polynomial degrees to test
  - The minimum alpha value to test
  - The maximum alpha value to test
  - The number of alpha values to test
  - 
 For the purposes of this project, the model types, data, number of K-Folds and a range of polynomial degrees to test were given. The alpha values were not.
 
 ## Identifying Optimal Alpha Ranges
 It is worth noting that Lasso, Ridge, and Elastic Net do not necisarially have to use the same alpha values. I used coefficient paths to estimate what a candidate Alpha range would be. This approach produced three coefficient paths.
  ### Ridge Regression
  ![image](https://user-images.githubusercontent.com/109169036/179863646-94f1f394-0210-4895-a6f5-f99d2c84b4fd.png)
  ### Lasso Regression
  ![image](https://user-images.githubusercontent.com/109169036/179863658-a755c265-5330-4fc9-8ad3-2ff2c09ade09.png)
  ### ElasticNet Regression
  ![image](https://user-images.githubusercontent.com/109169036/179870930-4987e5f7-c152-4965-a6b7-2c6a30b33e56.png)

These paths informed the alpha ranges I elected to test in the first trial

## First Trial
Given that the function I coded is not capable of exploring all three regression types at the same time, I had to call the function three times. 

For Ridge regression the function call was:
```Python
model=Ridge
k = 10
degrees = 3
a_start = 10e-5
a_end = 10e-5
a_tests = 2500

R2test, degree_value, a_value = Project2(model,X,y,k,degrees,a_start,a_end,a_tests)
```
For Lasso regression the function call was:
```Python
model = Lasso
k = 10
degrees = 3
a_start = 10e-5
a_end = 10e1
a_tests = 2500

R2test, degree_value, a_value = Project2(model,X,y,k,degrees,a_start,a_end,a_tests)
```
For ElasticNet regression the function call was:
```Python
model = ElasticNet
k = 10
degrees = 3
a_start = 10e-5
a_end = 10e1
a_tests = 2500

R2test, degree_value, a_value = Project2(model,X,y,k,degrees,a_start,a_end,a_tests)
```
## Trial One Results and Adjustments
After the function was ran for the three models, a function was printed for each one comparing the $R^{2}$ values to the alpha hyperparameter. The code also found the maximum $R^{2}$ score and the corresponding alpha value and the number of polynomial features. The code is:

```Python
idx_max = np.argmax(R2test)
print('Optimal Polynomal Degree:',degree_value[idx_max])
print('Optimal Alpha Value:',a_value[idx_max])
print('R^2 Value at that Point:',R2test[idx_max])

plt.figure(figsize=(8,4))
for d in range (1,degrees+1,1):
  plt.scatter(a_value[(d-1)*a_tests:d*a_tests],R2test[(d-1)*a_tests:d*a_tests],alpha = 0.5,label= 'Degree '+str(d)+' Polynomial', cmap = 'jet')
plt.xlabel('$\\alpha$')
plt.ylabel('Avg. $R^2$')
plt.title(model())
plt.legend()
plt.show()
```
It became apparent that al three initial alpha ranges could be improved.

- The optimal $R^{2}$ for Ridge was found at: alpha = 0.0001 ($10^{-5}$), Polynomial Features = 2. The $R^{2}$ was ≈ 0.5667
- The optimal $R^{2}$ for Lasso was found at: alpha ≈ 0.0240, Polynomial Features = 2. The $R^{2}$ was ≈ 0.5786
- The optimal $R^{2}$ for ElasticNet was found at: alpha ≈ 0.012, Polynomial Features = 2. The $R^{2}$ was ≈ 0.5726

These initial results suggested that the initial $R^{2}$ for Ridge was below $10^{-5}$, as $10^{-5}$ was the minimum value of alpha tested. Thereofore a second round of trials with a lower alpha hyperparameter would be necessary. Though both Lasso and ElasticNet yielded usable answeres, the precesiion could be increased.

## Second Trial
The Second Trial involved calling the same functions as in trial one, but with different alpha paramaters. 
- For Ridge, the starting alpha was decreased to ($10^{-7}$) while the ending alpha became 1. 
- For Lasso and Elastic Net, the starting alpha was kept at at ($10^{-5}$) and the ending alpha became 0.1 ($10^{-1}$).

## Trial Two Results
This trial found similar that the optimal $R^{2}$ value could be found at similar alpha values.
- The optimal $R^{2}$ for Ridge was found at: alpha ≈ 0.4045, Polynomial Features = 2. The $R^{2}$ was ≈ 0.5667
![image](https://user-images.githubusercontent.com/109169036/179868064-497a6fc2-649b-4508-9ad7-8003174606f3.png)
- The optimal $R^{2}$ for Lasso was found at: alpha ≈ 0.2345, Polynomial Features = 2. The $R^{2}$ was ≈ 0.5786
![image](https://user-images.githubusercontent.com/109169036/180067462-af9160af-ac3f-43e9-818c-74ca487ad304.png)
- The optimal $R^{2}$ for ElasticNet was found at: alpha ≈ 0.1369, Polynomial Features = 2. The $R^{2}$ was 0.5727
- ![image](https://user-images.githubusercontent.com/109169036/180067432-1de6bd60-824d-4865-91f6-d119e4c3010d.png)

The following graphs, which plot the average $R^{2}$ value as the alpha value changes, provide graphical evidence these $R^{2}$ values are indeed optimal.

### Ridge (The final alpha value tested was increased to show the downward slope.
![image](https://user-images.githubusercontent.com/109169036/180071265-d6b136a3-5831-4e30-a64b-fb7cfad20ea6.png)

### Lasso
![image](https://user-images.githubusercontent.com/109169036/180067900-aae42c94-3d70-41cf-aa67-773bd49d468e.png)

### Elastic Net
![image](https://user-images.githubusercontent.com/109169036/180067924-4deb0884-b9ba-4616-9625-d8b20953c1f5.png)

Based on the results from Trial Two,, I found that the optimal $R^{2}$ occured using the Lasso model, with a 2nd degree Polynomial and an alpha hyperparameter of 0.2345.

## Test for Normality
The last part of this project asked if the residuals of the optimal $R^{2}$ followed a normal distribution. I wrote the following function to check if the residuals followed a normal distribution in three ways. First, through a distributional plot. Second, through a quantile-quantile plot. Third, using the Kolmogorov-Smirnov and the Anderson-Darling Tests.

The function is as follows:
```Python
def Optimal_Residuals (model, X, y, a_value, degrees):
  import seaborn as sns
  from scipy import stats
  from scipy.stats import norm
  from sklearn.linear_model import Ridge
  from sklearn.linear_model import Lasso
  from sklearn.linear_model import ElasticNet
  from sklearn.pipeline import Pipeline

  scale = StandardScaler()
  
  poly = poly = PolynomialFeatures(degrees)
  pipe = Pipeline([['Scaler',scale],['Poly Feats',poly]])
  
  test_model = model (alpha = a_value, )

  poly_X = pipe.fit_transform(X)
  
  test_model.fit(poly_X,y)

  residuals = y - test_model.predict(poly_X)

  # Distributional Plot
  DP=plt.figure
  ax1 = sns.distplot(residuals,
                    kde=False,
                    color='deepskyblue',
                    hist_kws={"color":'green','ec':'black'},
                    fit=stats.norm,
                    fit_kws={"color":'red'})
  ax1.set(xlabel='Residuals', ylabel='Frequency')

  # Quantile-Quantile Plot
  import statsmodels.api as sm
  QQ=plt.figure
  sm.qqplot(residuals/np.std(residuals), loc = 0, scale = 1, line='s',alpha=0.5)
  plt.xlim([-2.5,2.5])
  plt.ylim([-2.5,2.5])
  plt.axes().set_aspect('equal')
  plt.grid(b=True,which='major', color ='grey', linestyle='-', alpha=0.5)
  plt.grid(b=True,which='minor', color ='grey', linestyle='--', alpha=0.15)
  plt.minorticks_on()

  # KS and Anderson Test
  dist = getattr(stats, 'norm')
  params = dist.fit(residuals)
  stats.kstest(residuals,'norm',params)
  stats.anderson(residuals,'norm')

  KS_Test = stats.kstest(residuals,'norm',params)
  AD_Test = stats.anderson(residuals,'norm')

  return DP, QQ, KS_Test, AD_Test
  ```
When the function was called using the optimal conditions found in the section "Trial Two Results" it returned the following information:
### Distribution Plot
![image](https://user-images.githubusercontent.com/109169036/180069251-5d694a4e-1634-4280-ac1a-de81f080dd5f.png)

### Quantile-Quantile Plot
![image](https://user-images.githubusercontent.com/109169036/180069286-0d8008b7-4c6e-49fc-ab1f-223740530457.png)

### Test Results
Kolmogorov-Smirnov Test Results: KstestResult(statistic=0.04924004730839393, pvalue=0.02365837157411001)

Anderson-Darling Test Results: AndersonResult(statistic=3.6735430141793586, critical_values=array([0.573, 0.653, 0.784, 0.914, 1.087]), significance_level=array([15. , 10. ,  5. ,  2.5,  1. ]))

This suggests that the residuals are not normally distrubuted.

## The Support Vector Regression Challenge
One additional challenge that my professor proposed was to write a function which uses Support Vector Regression to identify the optimal values for the C and epsilon hyperparamaters. For the purposes of this question, it was assumed that the user would input the desired kernal type and a single polynomial degree. This is primarially stipulated the limit the time that the code needs to run. The following code accomplishes the assigned task. 
```Python
def Project2_SVR (X, y, k, kernel_type, degree, C_start, C_end, C_tests, eps_start, eps_end, eps_tests, random_state=123):
  import numpy as np
  from sklearn.model_selection import KFold
  from sklearn.preprocessing import StandardScaler, PolynomialFeatures
  from sklearn.pipeline import Pipeline
  import matplotlib.pyplot as plt
  from sklearn.svm import SVR

  scale = StandardScaler()

  R2train_avg = []
  R2test_avg = []
  degree_value =[]
  c_value = []
  eps_value = []

  kf = KFold(n_splits=k, random_state=123, shuffle=True)
  
  poly_feat = d
  poly = PolynomialFeatures(degree = d)
  pipe = Pipeline([['Scaler',scale],['Poly Feats',poly]])

  for c in np.linspace(C_start,C_end,C_tests):
    reg_para = c
   
    for e in np.linspace (eps_start, eps_end, eps_tests):
      test_model = SVR(kernel = kernel_type, degree = poly_feat, C = reg_para, epsilon = e)
       
       R2train_raw = []
       R2test_raw = []
     
       for idxtrain, idxtest in kf.split(X):
          Xtrain = X[idxtrain]
          Xtest = X[idxtest]
          ytrain = y[idxtrain]
          ytest = y[idxtest]
          Xpolytrain = pipe.fit_transform(Xtrain)
          Xpolytest = pipe.transform(Xtest)

          #Regression
          test_model.fit(Xpolytrain,ytrain)
          R2train_raw.append(test_model.score(Xpolytrain,ytrain))
          R2test_raw.append(test_model.score(Xpolytest,ytest))
        R2test_avg.append(np.mean(R2test_raw))
        R2train_avg.append(np.mean(R2train_raw))
        degree_value.append(d)
        c_value.append(c)
        eps_value.append(e)
  
    return R2train_avg, R2test_avg, degree_value, c_value, eps_value
```
Calling the function with the following code
```Python
k = 10
kernel_type = 'linear'
degree = 1

C_start = 0.001
C_end = 10
C_tests = 10

eps_start = 0.001
eps_end = 10
eps_tests = 10

R2train, R2test, degree_value, C_value, EPS_value = Project2_SVR(X, y, k, kernel_type, degree, C_start, C_end, C_tests, eps_start,eps_end, eps_tests)

# SVR
idx_max = np.argmax(R2test)
print('Optimal Polynomal Degree:',degree_value[idx_max])
print('Optimal C Value:',C_value[idx_max])
print('Optimal Epsilon Value:',EPS_value[idx_max])
print('R^2 Value at that Point:',R2test[idx_max])
```
This code results in the optimal $R^{2}$ for a Degree 1 Support Vector Regression being found when: C ≈ 0.6131 and Epislon ≈. The $R^{2}$ is 0.5727 

Unfortunately, this function is incredibly slow and therefore it's application is limited without the use of more effecient methods of optimization (ie. grid search).
