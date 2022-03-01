# importing needed packages
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import statsmodels.api as sm

#importing datasets
df = pd.read_csv('DailyDelhiClimateTrain.csv', index_col=0, parse_dates=["date"])

# check if null values exists
df.info()

# plot the mean temperature
plt.figure(figsize = (18,8))
plt.plot(df['meantemp']);

# the function below considering the null hypothesis that data is not stationary
# and the alternate hypothesis that data is stationary
def adfuller_test(temp):
    result=adfuller(temp)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )

    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data is stationary")
    else:
        print("weak evidence against null hypothesis,indicating it is non-stationary ")

# apply the function
adfuller_test(df['meantemp'])

# adding seasonality
df['Temperature First Difference'] = df['meantemp'] - df['meantemp'].shift(1)
df['Seasonal First Difference']=df['meantemp']-df['meantemp'].shift(12)
df.head()

# testing if data is stationary
adfuller_test(df['Seasonal First Difference'].dropna())

# plot the mean seasonal
plt.figure(figsize = (18,8))
df['Seasonal First Difference'].plot();

# create auto-correlation
plt.figure(figsize = (18,8))
autocorrelation_plot(df['meantemp'])
plt.show()

fig = plt.figure(figsize=(18,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df['Seasonal First Difference'].dropna(),lags=60,ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df['Seasonal First Difference'].dropna(),lags=60,ax=ax2)

# building ARIMA model
model=ARIMA(df['meantemp'],order=(1,1,1))
model_fit=model.fit()
model_fit.summary()
