# TODO: Add import statements
import pandas as pd
from sklearn.linear_model import LinearRegression as lr
from matplotlib import pyplot as plt

# Assign the dataframe to this variable.
# TODO: Load the data
bmi_life_data = pd.read_csv("bmi_and_life_expectancy.csv")

print(bmi_life_data)
x = bmi_life_data[['BMI']]
y = bmi_life_data[['Life expectancy']]
# # Make and fit the linear regression model
# #TODO: Fit the model and Assign it to bmi_life_model
bmi_life_model = lr()
bmi_life_model.fit(x, y)

plt.scatter(x,y)
plt.plot(x, bmi_life_model.predict(x))
plt.show()
# # Mak a prediction using the model
# # TODO: Predict life expectancy for a BMI value of 21.07931
laos_life_exp = bmi_life_model.predict(21.07931)
print(laos_life_exp)
