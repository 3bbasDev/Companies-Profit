import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# import Dataset 
dataset = pd.read_csv('CompaniesProfit.csv')
X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:, -1].values



# df = pd.DataFrame(dataset,columns=['R&D Spend','Administration','Marketing Spend','State','Profit']) 
# plt.scatter(df['Administration'], df['Profit'], color='red')
# plt.title('Profit Price Vs Administration', fontsize=14)
# plt.xlabel('Administration', fontsize=14)
# plt.ylabel('Profit', fontsize=14)
# plt.grid(True)
# plt.show()

# Encodeing Company State from it to column with 0,1 values
#                                                    type encode ,column index
ColumnTrans = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])],remainder='passthrough')
X = np.array(ColumnTrans.fit_transform(X))

#X = X[:, 1:]

# Split dataset to train and test arraies Test = %33 , Train = %67
[X_Train,X_Test,Y_Train,Y_Test]= train_test_split(X,Y,test_size=0.33,random_state=0)

# print(X_Train[0])

# Train Model
Regression: object = LinearRegression()
Regression.fit(X_Train,Y_Train)


print('Intercept: \n', Regression.intercept_)
print('Coefficients: \n', Regression.coef_)

New_StatusCompany=1.0
New_RD_Spend=165349.2
New_Administration=136897.8
New_Marketing_Spend=471784.1

print ('Predicted Stock Index Price: \n', Regression.predict([[1.0,0.0,0.0,New_RD_Spend ,New_Administration,New_Marketing_Spend]]))

#print('Predicted Stock Index Price: \n',Regression.predict([[1.0,0.0,0.0,New_RD_Spend,New_Administration,New_Marketing_Spend]]))


# Predicting Test result
Y_Perdict: object = Regression.predict(X_Test)
np.set_printoptions(precision=2)
print(np.concatenate((Y_Perdict.reshape(len(Y_Perdict),1),Y_Test.reshape(len(Y_Test),1)),1))


import statsmodels.api as sm
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
print(regressor_OLS.summary())
X_opt = X[:, [0, 1, 3, 4, 5]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
print(regressor_OLS.summary())
X_opt = X[:, [0, 3, 4, 5]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
print(regressor_OLS.summary())
X_opt = X[:, [0, 3, 5]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
print(regressor_OLS.summary())
X_opt = X[:, [0, 3]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
print(regressor_OLS.summary())


