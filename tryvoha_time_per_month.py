import datetime
from tokenize import String
from datetime import date, datetime
import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from typing import List, Tuple, Dict, Any
from datetime import datetime, timedelta
from kyiv_data import tryvoha_kyiv_data
import json
import pytz
import datetime
from tokenize import String
from datetime import date, datetime
import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from typing import List, Tuple, Dict, Any
from datetime import datetime, timedelta
import json
import pytz
from tryvoha_parser import differenceMoscowUkraine, is_alarm_on, ukraineToMoscowTime

minutes_per_interval = 60
threshold = 0
print("Minutes per interval: ", minutes_per_interval)
print("Threshold: ", threshold)

start_date = ukraineToMoscowTime(datetime(2022, 2, 1, 0, 0, 0))
print("Since [Moscow time]: ", start_date)

end_date = ukraineToMoscowTime(datetime(2024, 3, 1, 0, 0, 0))
print("Until [Moscow time]: ", end_date)


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
data['Status'] = [is_alarm_on(timestamp) for timestamp in timestamp_array]

def average_per_month(data):
    data['Time'] = pd.to_datetime(data['Time'])
    data = data.set_index('Time')
    data = data.resample('ME').mean()
    return data

average_per_month = average_per_month(pd.DataFrame(data))
print(average_per_month)

# write to file inluding Time column
average_per_month.to_csv('tryvoha_time_per_month.csv')