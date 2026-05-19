import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error



data = pd.read_csv("C:/AED_Waiting_Time.csv")
print(data.head())
data['Arrival_Time'] = pd.to_datetime(data['Arrival_Time'])
data['Arrival_Time'] = data['Arrival_Time'].dt.hour * 60 + data['Arrival_Time'].dt.minute
print(data.dtypes)


X = data[['Patient Age','Gender_Code','Arrival_Time','Day_Code',
          'Emergency_Type_Code','Triage_Level',
          'Queue_Length','Doctors_Available',
          'Nurses_Available','Department_Code']]

y = data['Patient Waittime']



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
model.fit(X_train, y_train)




predictions = model.predict(X_test)


from sklearn.metrics import mean_absolute_error

error = mean_absolute_error(y_test, predictions)
print("Mean Absolute Error:", error)




from sklearn.metrics import r2_score
print(r2_score(y_test, predictions))



############ linear regression#########
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

lr = LinearRegression()
lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

print("Linear Regression MAE:", mean_absolute_error(y_test, lr_pred))
print("Linear Regression R2:", r2_score(y_test, lr_pred))





######## Support Vector Regression######
from sklearn.svm import SVR

svr = SVR()
svr.fit(X_train, y_train)

svr_pred = svr.predict(X_test)

print("SVR MAE:", mean_absolute_error(y_test, svr_pred))
print("SVR R2:", r2_score(y_test, svr_pred))


pip install xgboost

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

xgb = XGBRegressor()

xgb.fit(X_train, y_train)

xgb_pred = xgb.predict(X_test)

print("XGBoost MAE:", mean_absolute_error(y_test, xgb_pred))
print("XGBoost R2:", r2_score(y_test, xgb_pred))



#####catboost algorithm####
pip install catboost


from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Initialize CatBoost Regressor
cat = CatBoostRegressor(verbose=0)  # verbose=0 to suppress output

# Fit the model
cat.fit(X_train, y_train)

# Make predictions
cat_pred = cat.predict(X_test)

# Evaluate
print("CatBoost MAE:", mean_absolute_error(y_test, cat_pred))
print("CatBoost R2:", r2_score(y_test, cat_pred))







#### MAE visualization####
import matplotlib.pyplot as plt

algorithms = ['Linear Regression','CART','Random Forest','XGBoost','CatBoost','SVR']
mae_values = [6.64, 3.85, 2.86, 2.95, 2.76, 9.47]

plt.bar(algorithms, mae_values, color=['blue','orange','green','red','purple','gray'])
plt.ylabel("Mean Absolute Error (minutes)")
plt.title("Algorithm MAE Comparison")
plt.xticks(rotation=20)
plt.show()

###########R2 SCORE#####
r2_scores = [0.622,0.889,0.9449,0.9388,0.9483,-0.085]

plt.bar(algorithms, r2_scores, color=['blue','orange','green','red','purple','gray'])
plt.ylabel("R² Score")
plt.title("Algorithm R² Comparison")
plt.xticks(rotation=20)
plt.show()




#############Random Forest or CatBoost, you can show which factors affect waiting time most:
importances = model.feature_importances_  # if model is RandomForest
features = X.columns

plt.figure(figsize=(10,5))
plt.barh(features, importances, color='teal')
plt.xlabel("Importance")
plt.title("Feature Importance - Random Forest")
plt.show()



##########WAITING TIME DISTIBUTUION#########
plt.hist(data['Patient Waittime'], bins=20, color='skyblue')
plt.xlabel("Waiting Time (minutes)")
plt.ylabel("Number of Patients")
plt.title("Distribution of Waiting Time")
plt.show()
##########Triage Level vs Waiting Time:###
import seaborn as sns
sns.boxplot(x='Triage_Level', y='Patient Waittime', data=data)
plt.title("Triage Level vs Waiting Time")
plt.show()


###########Queue Length vs Waiting Time:
sns.scatterplot(x='Queue_Length', y='Patient Waittime', data=data)
plt.title("Queue Length vs Waiting Time")
plt.show()





##########Step 1: Prepare the new patient data######
# New patient example
new_patient = {
    'Patient Age': 35,
    'Gender_Code': 0,        # 0=M, 1=F
    'Arrival_Time': 8*60 + 30,  # 8:30 AM → 510 minutes
    'Day_Code': 2,            # 2=Tuesday
    'Emergency_Type_Code': 1, # Heart Attack
    'Triage_Level': 1,        # Critical
    'Queue_Length': 10,
    'Doctors_Available': 3,
    'Nurses_Available': 5,
    'Department_Code': 1
}

import pandas as pd

new_patient_df = pd.DataFrame([new_patient])


# Assuming your CatBoost model is called 'cat'
predicted_wait = cat.predict(new_patient_df)

print(f"Predicted waiting time: {predicted_wait[0]:.2f} minutes")


#########remaining algorithms###
# Linear Regression
lr_pred = lr.predict(new_patient_df)
print("Linear Regression Prediction:", lr_pred[0])


# Random Forest
rf_pred = rf.predict(new_patient_df)
print("Random Forest Prediction:", rf_pred[0])

# XGBoost
xgb_pred = xgb.predict(new_patient_df)
print("XGBoost Prediction:", xgb_pred[0])

# CatBoost
cat_pred = cat.predict(new_patient_df)
print("CatBoost Prediction:", cat_pred[0])

# SVR
svr_pred = svr.predict(new_patient_df)
print("SVR Prediction:", svr_pred[0])

# Linear Regression
lr_pred = lr.predict(X_test)

# Random Forest
rf_pred = rf.predict(X_test)

# XGBoost
xgb_pred = xgb.predict(X_test)

# CatBoost
cat_pred = cat.predict(X_test)

# Mean of actual waiting times
mean_wait = y_test.mean()

# Linear Regression Accuracy
lr_acc = (1 - mean_absolute_error(y_test, lr_pred)/mean_wait)*100
print(f"Linear Regression Accuracy: {lr_acc:.2f}%")

# Random Forest Accuracy
rf_acc = (1 - mean_absolute_error(y_test, rf_pred)/mean_wait)*100
print(f"Random Forest Accuracy: {rf_acc:.2f}%")

# XGBoost Accuracy
xgb_acc = (1 - mean_absolute_error(y_test, xgb_pred)/mean_wait)*100
print(f"XGBoost Accuracy: {xgb_acc:.2f}%")

# CatBoost Accuracy
cat_acc = (1 - mean_absolute_error(y_test, cat_pred)/mean_wait)*100
print(f"CatBoost Accuracy: {cat_acc:.2f}%")





#############features importance#############
import pandas as pd

# Get feature importance from Random Forest
importance = rf.feature_importances_

# Create dataframe
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importance
})

# Sort values
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

print(feature_importance)

# Save to CSV for Power BI
feature_importance.to_csv("feature_importance.csv", index=False)
import matplotlib.pyplot as plt

# Get feature importance
importance = rf.feature_importances_
features = X.columns

# Plot graph
plt.figure(figsize=(8,5))
plt.barh(features, importance)

plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance Affecting Waiting Time")

plt.show()

