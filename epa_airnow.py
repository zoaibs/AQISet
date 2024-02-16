import requests
from datetime import datetime

key = 'D51D9F5E-2414-4B24-A464-195CFB5BC484'
parameters = 'O3,pm25,pm10,co,no2,so2' #pollutants
datatype = 'A' #aqi
form = 'application/json'
verbose = '0'
includerawconcentrations = '0'

def get_aqi(lat, lon, time):
    #111km = 1 lat/long
    #10km = 0.09 lat/long
    #params for bbox
    lat_min = lat - 0.045
    lat_max = lat + 0.045
    lon_min = lon - 0.045
    lon_max = lon + 0.045
    bbox = f'{lon_min},{lat_min},{lon_max},{lat_max}'
    window = 60 * 60 * 24 * 1#how many days
    
    startdate = datetime.fromtimestamp(time).strftime(f"%Y-%m-%dT%H:%M")
    enddate = datetime.fromtimestamp(time+window).strftime(f"%Y-%m-%dT%H:%M")
    print(startdate, enddate)
    response = requests.get(f'https://www.airnowapi.org/aq/data/?startDate={startdate}&endDate={enddate}&parameters=OZONE,PM25,PM10,CO,NO2,SO2&BBOX={bbox}&dataType=A&format={form}&verbose=0&monitorType=0&includerawconcentrations=0&API_KEY=D51D9F5E-2414-4B24-A464-195CFB5BC484')
    data = response.json()
    print(data)
get_aqi(30, -95, 1610000000)