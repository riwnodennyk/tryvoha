import datetime
from tokenize import String
from datetime import date, datetime
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from typing import List, Tuple, Dict, Any
from datetime import datetime, timedelta
from file1 import dataSetString
import json

look_back = relativedelta(weeks=8)

print("Look back: ", look_back)

def calculateDaysBetweenDates(begin, end):
    return (end - begin).days

def parseDataSetIntoExpectedFormat(dataSet):
    result = []
    for line in dataSet.split("\n"):
        if len(line) > 0:
            time, status, *duration = line.split("\t")
            result.append((time, status, duration))
    return result


def parsedDataSplitIntoLines(parsedData):
    result = []
    for line in parsedData:
        if len(line) > 0:
            time, status, duration = line
            result.append((time, status, duration))
            result.append("\n")
    return result

allAlerts = parseDataSetIntoExpectedFormat(dataSetString)

def is_alarm_on(timestamp, data = allAlerts):
    for record in data:
        time, status, duration = record
        record_time = datetime.strptime(time, "%H:%M %d.%m.%y")

        if timestamp >= record_time:
            return status == "🔴 Повітряна тривога!"

    return None  # Return None if the timestamp is not found in the data

# Example usage:
timestamp_to_check = datetime(2023, 12, 19, 23, 59, 0)
result = is_alarm_on(timestamp_to_check)


end_date = datetime.strptime(allAlerts[0][0], "%H:%M %d.%m.%y")
print("Until: ", end_date)

start_date = end_date - look_back
print("Since: ", start_date)

minutes_per_interval = 5

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
labels = [is_alarm_on(timestamp) for timestamp in timestamp_array]
data['Status'] = labels

# Create a DataFrame
df = pd.DataFrame(data)

def feature_engineering(df):
    df['Hour'] = df['Time'].dt.hour
    df['DayOfWeek'] = df['Time'].dt.dayofweek + 1  # Adding 1 to match the range (1-7)
    df['DayOfYear'] = df['Time'].dt.day_of_year
    df['Year'] = df['Time'].dt.year
    # df['Weight'] = 2024*365 - df['Year']*365-df['DayOfYear']  # Adjust the weighting based on your data
    return df

df = feature_engineering(df)
# Display the constructed labeled data
# print(df)

# Selecting features and target variable
X = df[['Hour', 'DayOfWeek'
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

# Create a DataFrame with all hours of all days of the week
all_hours = range(24)
all_days = range(1, 8)  # Assuming dayofweek range is 1-7

all_combinations = [(hour, day) for hour in all_hours for day in all_days]
all_timestamps = [end_date + timedelta(hours=hour, days=day-1) for hour, day in all_combinations]

all_data = {'Time': all_timestamps}
all_df = pd.DataFrame(all_data)

all_df = feature_engineering(all_df)

# Making predictions (probabilities) for all hours of all days
all_probabilities = model.predict_proba(all_df[['Hour', 'DayOfWeek'
                                                # , 'Weight'
                                                ]])

# Adding prediction probabilities to the DataFrame
for i, class_label in enumerate(model.classes_):
    all_df[f'Probability_{class_label}'] = all_probabilities[:, i]

# Sort the DataFrame by 'DayOfWeek' and then by 'Hour'
all_df.sort_values(by=['DayOfWeek', 'Hour'], inplace=True)

# Display the sorted DataFrame with prediction probabilities for all hours of all days
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

selected_columns = ['Hour', 'DayOfWeek', 'Probability_True']
csv_data = all_df[selected_columns]

def convertCsvDataToJson(csv_data):
    result = []
    for index, row in csv_data.iterrows():
        result.append({
            "Hour": row["Hour"],
            "DayOfWeek": row["DayOfWeek"],
            "Probability_True": row["Probability_True"]
        })
    return result

def writeJsonIntoFile(json_data):
    with open('data.json', 'w') as f:
        f.write(json.dumps(json_data, indent=2))

# Assuming you have a DataFrame named 'csv_data'


writeJsonIntoFile(convertCsvDataToJson(csv_data))
#print(csv_data)

# Reset options to default after printing
pd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')

# Evaluating the model


threshold = 0
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

conf_matrix = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{conf_matrix}")