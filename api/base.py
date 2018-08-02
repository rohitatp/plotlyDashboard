import urllib.request, urllib.error, urllib.parse, json, datetime, pytz, calendar, os
import pandas, numpy
import pickle as pickle

# import time, datetime, pandas, numpy

def get_local_datetime(timestamp, timezone):
    utc_dt = pytz.utc.localize(datetime.datetime.utcfromtimestamp(timestamp))
    return utc_dt.astimezone(pytz.timezone(timezone))
