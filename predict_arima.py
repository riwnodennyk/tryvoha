import datetime
import itertools
from tokenize import String
from datetime import date, datetime
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from typing import List, Tuple, Dict, Any
from datetime import datetime, timedelta
from kyiv_data import tryvoha_kyiv_data
from statsmodels.tsa.arima.model import ARIMA
import json
import pytz
import pandas as pd
import numpy as np
from datetime import timedelta
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

minutes_per_interval = 30
threshold = 0
look_back = relativedelta(weeks=3)
print("Minutes per interval: ", minutes_per_interval)
print("Threshold: ", threshold)
print("Look back: ", look_back)

now = datetime.now()
ukraine_tz = pytz.timezone("Europe/Kiev")
moscow_tz = pytz.timezone("Europe/Moscow")


def calculateDaysBetweenDates(begin, end):
    return (end - begin).days

def parseDataSetIntoExpectedFormat(dataSet):
    result = []
    for line in dataSet.split("\n"):
        if len(line) > 0:
            time, status, *duration = line.split("\t")
            result.append((time, status))
    return result


def parsedDataSplitIntoLines(parsedData):
    result = []
    for line in parsedData:
        if len(line) > 0:
            time, status, duration = line
            result.append((time, status, duration))
            result.append("\n")
    return result

allAlerts = parseDataSetIntoExpectedFormat(tryvoha_kyiv_data)

df = pd.DataFrame(allAlerts)
df.columns = ['datetime', 'status']

# Drop duplicates based on 'datetime'
df = df.drop_duplicates('datetime')

# Preprocessing the data
df['datetime'] = pd.to_datetime(df['datetime'], format='%H:%M %d.%m.%y')
df.set_index('datetime', inplace=True)

# Resampling the data to hourly frequency and filling missing values
df = df.resample('H').ffill()

df = df[df.index >= now - look_back]

df['hour_of_day'] = df.index.hour
df['day_of_week'] = df.index.dayofweek

# Extracting relevant features for training

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

print(df)

# Reset options to default after printing
pd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')

df['status'] = df['status'].apply(lambda x: 0 if x == '🟢 Відбій тривоги' else 1)

# Extracting relevant features for training
print(df)

# Train model on df

df_resampled = df


# Split the data into training and testing sets
train_size = int(len(df_resampled) * 0.8)
train_data, test_data = df_resampled[:train_size], df_resampled[train_size:]

model = SARIMAX(df['status'], exog=df[['hour_of_day', 'day_of_week']], order=(1, 1, 1), seasonal_order=(1, 1, 1, 24*7))

model_fit = model.fit()

# Making predictions for one week into the future
future_date = df.index[-1] + timedelta(hours=1)
future_prediction = model_fit.get_forecast(steps=24*7)  # 24 steps for 1 day
forecast_index = pd.date_range(start=future_date, periods=24*7, freq='H')
forecast_series = pd.Series(future_prediction.predicted_mean.values, index=forecast_index)

# Printing the forecasted values
print("Forecasted Values:")
print(forecast_series)

# Convert the forecasted values to a DataFrame
forecast_series = forecast_series.reset_index()

forecast_series.columns = ['datetime', 'probability']

def convertCsvDataToJson(csv_data):
    result = []
    for index, row in csv_data.iterrows():
        result.append({
            "Hour": row["datetime"].hour,
            "DayOfWeek": row["datetime"].dayofweek + 1,
            "Probability_True": row["probability"]
        })
    return result

def writeJsonIntoFile(json_data):
    with open('data.json', 'w') as f:
        f.write(json.dumps(json_data, indent=2))

writeJsonIntoFile(convertCsvDataToJson(forecast_series))

# Split the data into training and testing sets
train_size = int(len(train_data) * 0.8)
train, test = train_data[:train_size], train_data[train_size:]

# Define the range of values for p, d, and q
p_values = range(0, 6)
d_values = range(0, 2)
q_values = range(0, 6)

best_order = None
best_rmse = float('inf')

# Iterate over all combinations and find the best order
for order in itertools.product(p_values, d_values, q_values):
    try:
        model = ARIMA(train, order=order)
        model_fit = model.fit()
        predictions = model_fit.forecast(steps=len(test))[0]

        # Calculate RMSE (Root Mean Squared Error)
        rmse = np.sqrt(mean_squared_error(test, predictions))

        # Update best order if the current one is better
        if rmse < best_rmse:
            best_rmse = rmse
            best_order = order

    except Exception as e:
        continue

print(f"Best Order: {best_order}, Best RMSE: {best_rmse}")

# Training the ARIMA model
# train_data = train_data.apply(pd.to_numeric, errors='coerce')
model = ARIMA(train_data, order=(5, 1, 0))  
# Adjust order based on data characteristics
model_fit = model.fit()

# Making predictions for one day into the future
future_date = df.index[-1] + timedelta(hours=1)
future_prediction = model_fit.get_forecast(steps=24)  # 24 steps for 1 day
forecast_index = pd.date_range(start=future_date, periods=24, freq='H')
forecast_series = pd.Series(future_prediction.predicted_mean.values, index=forecast_index)

# Printing the forecasted values
print("Forecasted Values:")
print(forecast_series)

def differenceBetweenUkraineAndMoscowTimeZonesInHours(when):
    return int((moscow_tz.localize(when)-ukraine_tz.localize(when)).total_seconds() / 3600)

def is_alarm_on(timestamp, data = allAlerts):
    for record in data:
        time, status, duration = record
        record_time = datetime.strptime(time, "%H:%M %d.%m.%y")

        if timestamp >= ukraineToMoscowTime(record_time):
            # Return None if the timestamp is not found in the data
            return status == "🔴 Повітряна тривога!"

    return None 

def ukraineToMoscowTime(time):
    return time - timedelta(hours=differenceBetweenUkraineAndMoscowTimeZonesInHours(time)) 

# Example usage:
timestamp_to_check = datetime(2023, 12, 19, 23, 59, 0)
result = is_alarm_on(timestamp_to_check)


end_date = ukraineToMoscowTime(now).replace(minute=0, second=0, microsecond= 0)
print("Until [Moscow time]: ", end_date)

start_date = end_date - look_back
print("Since [Moscow time]: ", start_date)


def timestamps(start_date, end_date):
    current_date = start_date
    timestamp_array = []

    while current_date <= end_date:
        timestamp_array.append(current_date)
        current_date += timedelta(minutes=minutes_per_interval)

    return timestamp_array

timestamp_array = timestamps(start_date, end_date)

# Construct labeled data
data = {'Time': timestamp_array}
scanForBeforeAndAfter = True
print("Scan for before and after: ", scanForBeforeAndAfter)

if(scanForBeforeAndAfter):
    labels = [is_alarm_on(timestamp) or is_alarm_on(timestamp + timedelta(minutes=minutes_per_interval)) or is_alarm_on(timestamp - timedelta(minutes=minutes_per_interval)) for timestamp in timestamp_array]
else:
    labels = [is_alarm_on(timestamp) for timestamp in timestamp_array]
data['Status'] = labels

# Create a DataFrame
df = pd.DataFrame(data)

def feature_engineering(df):
    df['Moscow_Hour'] = df['Time'].dt.hour
    df['Moscow_DayOfWeek'] = df['Time'].dt.dayofweek + 1  # Adding 1 to match the range (1-7)
    # df['DayOfYear'] = df['Time'].dt.day_of_year
    # df['Year'] = df['Time'].dt.year
    # df['Weight'] = 2024*365 - df['Year']*365-df['DayOfYear']  # Adjust the weighting based on your data
    return df

df = feature_engineering(df)
# Display the constructed labeled data
# print(df)

# Selecting features and target variable
X = df[['Moscow_Hour', 'Moscow_DayOfWeek'
        # , 'Weight'
        ]]
y = df['Status']

# Splitting the data into training and testing sets
one_week = 7*24*int(60/minutes_per_interval)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=one_week, random_state=42, shuffle=False)

# Training a RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# print(f"X_train: {X_train}")
# print(f"X_test: {X_test}")

# Making predictions on the test set
y_pred = model.predict(X_test)

def evaluateModel():
    # Evaluating the model
    # print(f"X_test: {X_test}")
    probabilities = model.predict_proba(X_test)[:, 1]  # Assuming positive class is at index 1
    # print(f"Probabilities: {probabilities}")
    predictions = (probabilities > threshold).astype(int)

    acc = accuracy_score(y_test, predictions)
    print(f"Accuracy: {acc:.2%}")
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F1 Score: {f1:.2%}")

    conf_matrix = confusion_matrix(y_test, predictions)
    print(f"Confusion Matrix:\n{conf_matrix}")
    
evaluateModel()

# nan_indices = np.isnan(X)
# print("Indices with NaN values:", np.where(nan_indices))

# # Retraining the model on the entire data
# model.fit(X, y)

# Create a DataFrame with all hours of all days of the week
all_hours = range(24)
all_days = range(1, 8)  # Assuming dayofweek range is 1-7

all_combinations = [(hour, day) for hour in all_hours for day in all_days]
all_timestamps = [end_date + timedelta(hours=hour, days=day-1) for hour, day in all_combinations]

all_data = {'Time': all_timestamps}
all_df = pd.DataFrame(all_data)

all_df = feature_engineering(all_df)

# Making predictions (probabilities) for all hours of all days
all_probabilities = model.predict_proba(all_df[['Moscow_Hour', 'Moscow_DayOfWeek'
                                                # , 'Weight'
                                                ]])

# Adding prediction probabilities to the DataFrame
for i, class_label in enumerate(model.classes_):
    all_df[f'Probability_{class_label}'] = all_probabilities[:, i]

# Sort the DataFrame by 'DayOfWeek' and then by 'Hour'
all_df.sort_values(by=['Moscow_DayOfWeek', 'Moscow_Hour'], inplace=True)

# Display the sorted DataFrame with prediction probabilities for all hours of all days
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

#print(csv_data)

# Reset options to default after printing
pd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')
