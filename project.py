import streamlit as st
import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error

np.random.seed(123)

def scatterplot(x,y, label):
    plt.scatter(x, y, color='r', label=label)
    return plt

# We used these two datasets from Kaggle.
df1 = pd.read_csv("world-happiness-report.csv")
df2= pd.read_csv('world-happiness-report-2021.csv')

# Once we read in the data, we dropped any columns that were not mutually shared between both datasets.
df2.drop(['Regional indicator', 'Standard error of ladder score', 'upperwhisker', 'lowerwhisker',"Ladder score in Dystopia", "Explained by: Log GDP per capita", "Explained by: Social support", "Explained by: Healthy life expectancy", "Explained by: Freedom to make life choices", "Explained by: Generosity","Explained by: Perceptions of corruption","Dystopia + residual"], axis = 1, inplace = True)
df1.drop(['Positive affect', 'Negative affect'], axis=1, inplace=True)
# We renamed these columns in dataframe two to make them have the exact titles that dataframe 1 used. This will help us keep our data aligned.
df2.rename(columns={'Logged GDP per capita': 'Log GDP per capita', 'Ladder score': 'Life Ladder', 'Healthy life expectancy':'Healthy life expectancy at birth'}, inplace= True)

# We want to combine the two data frames. We first added a year column to dataframe 2 to make it exactly like dataframe 1. Then we are append dataframe 2 to dataframe 1.
# Then we grouped by the country name and averaged all the values to create a dataframe with one row for each country to perform analysis. Any cell that is empty we filled with the average value of the entire column.
df2["year"] = 2021
df = df1.append(df2)
df.drop(['year'], axis=1, inplace=True)
df = df.groupby("Country name").mean()
df.fillna(df.mean(), inplace=True)

# In the dashboard's pairplot, we see that Perceptions of corruptions is left skewed therefore, to normalize the variable we performed a squared transformation.
df["Perceptions of corruption squared"] = df["Perceptions of corruption"]**2

# We want to perform data visualization to see relationships, correlations, and patterns in the data.
st.title("World Happiness Report Analysis Dashboard")
st.header("By: Elise Pereira and Peter Blust")

# The first graph we created is a scatter plot. We allowed the user to choose which variable to analyze and see all of the variables on one plot.
st.sidebar.header("Inputs for Correlation Scatter Plot")
variable = st.sidebar.radio("Choose Predictor Variable:", ["Log GDP per capita", "Social support", "Healthy life expectancy at birth", "Freedom to make life choices", "Generosity", "Perceptions of corruption squared"])

st.header("Scatter Plot")
st.write("This graph shows the relationship between the dependent variable (happiness score) and the predictor variables. The user can choose as one "
             "variable at a time to analyze on the chart")
fig, ax = plt.subplots()
scatterplot(df[variable], df['Life Ladder'], variable)
ax.legend()
plt.title(f"Relationship between Life Ladder and {variable}")
plt.xlabel(f"{variable}")
plt.ylabel("Life Ladder")
st.pyplot(plt)
# The scatter plot shows that all of the variables besides Generosity are linearly related. To further analyze correlations and examine normality,
# we created a pairplot that shows correlation graphs between all the variables as well as histograms.

st.header("Pair Plot")
st.write("We can visualize all of the correlations between the predictor variables as well as examine the distributions of each variable through a histogram.")
fig = sns.pairplot(df)
st.pyplot(fig, hue="species")

# as mentioned earlier, the perception of corruption variable was left skewed, so we transformed it. Now, all of the variables are
# approximately normally distributed.

# We created another visual showing the relationship between life expectancy and life ladder, with the marker size being the Log GDP value.
# These variables had a very strong relationship so we wanted to further visualize that.
fig, ax = plt.subplots(1,1, figsize=(10, 5))
st.header("Scatter Plot")
st.write("This scatter plot shows the relationship between life expectancy, log GDP per capita, and Life Ladder. The size of the marker"
         "corresponds to the Log GDP per capita.")
sns.scatterplot(data=df, x='Healthy life expectancy at birth', y='Life Ladder', alpha=0.9,ec='black',size=df["Log GDP per capita"]*100000, legend=True, sizes=(5, 500))
ax.set_xlabel("Life Expectancy")
ax.set_ylabel("Happiness Index Score")
ax.tick_params(axis = 'both', which = 'major', labelsize = 10)
ax.legend(loc="upper left")
st.pyplot(fig)
# This shows us further about the linear relationship between these 3 variables because as life expectancy increases so does the size of the marker.

# due to the linear relationships between the variables, we were motivated to perform linear regression to predict
# happiness scores from predictor variables.
# To begin, we split the data into training and testing dataframes, using an 80/20 split.
rows_for_training = np.random.choice(df.index,133, False)
training = df.index.isin(rows_for_training)
df_train = df[training]
df_test = df[~training]

# We first included all of the variables in our multiple linear regression.
x_train = df_train[["Log GDP per capita", "Social support", "Healthy life expectancy at birth", "Freedom to make life choices", "Generosity", "Perceptions of corruption squared"]]
y_train = df_train["Life Ladder"]
x_train = sm.add_constant(x_train)

# We fit the model to our training data using ordinary least squares regression
model = sm.OLS(y_train, x_train)
results = model.fit()
print(results.summary())
# As seen in the results, generosity is not significant, which makes sense because it was not linearly correlated with the happiness index score.
# We took out the generosity variable and performed our regression again.
x_train = df_train[["Log GDP per capita", "Social support", "Healthy life expectancy at birth", "Freedom to make life choices", "Perceptions of corruption squared"]]
y_train = df_train["Life Ladder"]

x_train = sm.add_constant(x_train)
new_model = sm.OLS(y_train, x_train)
new_results = new_model.fit()
print(new_results.summary())
# As seen in the results, our model is significant and is useful based on the low p-value of the F-statistic.
# Also, the adjusted R^2 is 80.3% which is very high.

# Now that we have a good model, we want to test the model for accuracy using the testing data. We only included the variables in the model.
x_test = df_test[["Log GDP per capita", "Social support", "Healthy life expectancy at birth", "Freedom to make life choices", "Perceptions of corruption squared"]]
y_test = df_test["Life Ladder"]
x_test = sm.add_constant(x_test)
predictions = new_results.predict(x_test)

# To analyze the goodness of fit of our model, we plotted the residuals and calculated the mean-squared error.
residuals = y_test - predictions
fig, ax = plt.subplots(1,1, figsize=(10, 5))
ax.scatter(predictions, residuals)
ax.set_xlabel("Predicted Values")
ax.set_ylabel("Residuals")
st.header("Residual Plot")
st.write("After creating our model, we plotted the residuals. As shown below, the residuals are evenly distributed around 0 "
         "and there is no clear patterns.")

st.pyplot(fig)
mse = mean_squared_error(y_test,predictions)
print(f'MSE: {mse}')
print(f'RMSE : {np.sqrt(mse)}')

# our MSE and RMSE are very close to 0 and our residuals appear approximately distributed around 0 with no apparent pattern.
# Based on these results, our model seems to be a good model to predict world happiness scores in the future.
