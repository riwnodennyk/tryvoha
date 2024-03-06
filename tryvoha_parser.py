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
from file1 import dataSetString
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

def differenceMoscowUkraine(when):
    ukraine_tz = pytz.timezone("Europe/Kiev")
    moscow_tz = pytz.timezone("Europe/Moscow")
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
    return time - timedelta(hours=differenceMoscowUkraine(time)) 
