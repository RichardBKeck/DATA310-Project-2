# DATA310-Project-2

This project was compleated as a part of DATA 310 at William & Mary. The project consisted of one programming assignments. In addition to the README file, this repository contains a folder which has the .ipynb files for all questions.

### Question One

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

The next step I took was to write a function that could be used to test and identify any combination of polynomial features,linear models, and alpha parameters. I did so using a series of "for loops and an internal K-Fold cross validation. The code that I produced is below.

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
      test_model = model(alpha= a, max_iter=5000)

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
  
  return R2 train_avg, R2test_avg, degree_value, a_value
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
 For the purposes of this project, the model types, data, K-Fold splits and polynomial degrees to test were given. The alpha values were not.
 
 ### Identifying Optimal Alpha Ranges
 It is worth noting that Lasso, Ridge, and Elastic Net do not necisarially have to use the same alpha values. I used coefficient paths to estimate what a candidate Alpha range would be. This approach produced three coefficient paths.
  ![image](https://user-images.githubusercontent.com/109169036/179863646-94f1f394-0210-4895-a6f5-f99d2c84b4fd.png)
  ![image](https://user-images.githubusercontent.com/109169036/179863658-a755c265-5330-4fc9-8ad3-2ff2c09ade09.png)
  ![image](https://user-images.githubusercontent.com/109169036/179863815-cf8b31b5-2a96-4caa-97b7-230f5212818c.png)

These paths led me to estimate that the optimal alpha hyperparameter for **Ridge regression** would be somewhere between $10^{0}$ and $10^{5}$, that the optimal alpha hyperparameter for **Lasso regression** and **ElasticNet Regression** would be somewhere between $10^{-3}$ and $10^{1}$

### First Function Calls
Given that the function I coded is not capable of exploring all three regression types at the same time, I had to call the function three times. 

For Ridge regression the function call was:
```Python
model=Ridge
k = 10
degrees = 3
a_start = 1
a_end = 100,000
a_tests = 2000

R2test, degree_value, a_value = Project2(model,X,y,k,degrees,a_start,a_end,a_tests)
```
For Lasso regression the function call was:
```Python
model = Lasso
k = 10
degrees = 3
a_start = 0.001
a_end = 10
a_tests = 2000

R2test, degree_value, a_value = Project2(model,X,y,k,degrees,a_start,a_end,a_tests)
```
For ElasticNet regression the function call was:
```Python
model = ElasticNet
k = 10
degrees = 3
a_start = 0.001
a_end = 10
a_tests = 2000

R2test, degree_value, a_value = Project2(model,X,y,k,degrees,a_start,a_end,a_tests)
```
### Initial Results and Adjustments
