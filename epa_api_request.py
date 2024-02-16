import requests
import datetime

api_key = '5e12c5066c174ebe2352f9906c5296da' 
api_key_temp = 'f6e3c7874e0739ffedb69b8663c3981a'
epa_key = 'bluekit96'
begin = 1609477200
ppbs = {}
#f = open("aqi_stats.txt", 'w')

## PM2.5 Sub-Index calculation
def get_pm2_5_aqi(x):
    if x <= 12.0:
        return 50 / 12 * x
    elif x <= 35.4:
        return (100 - 51) / (35.4 - 12.1) * (x - 12.1) + 51
    elif x <= 55.4:
        return (101 - 150) / (55.4 - 35.5) * (x - 35.5) + 101
    elif x <= 150.4:
        return (151 - 200) / (150.4 - 55.5) * (x - 55.5) + 151
    elif x <= 250.4:
        return (300 - 201) / (250.4 - 150.5) * (x - 150.5) + 201
    elif x <= 350.4:
        return (400 - 301) / (350.4 - 250.5) * (x - 250.5) + 301
    elif x > 350.4:
        return (500 - 401) / (500.4 - 350.5) * (x - 350.5) + 401 
    else:
        return 0

## PM10 Sub-Index calculation
def get_pm10_aqi(x):
    if x <= 54.0:
        return (50 - 0) / (54 - 0) * (x - 0) + 0
    elif x <= 154:
        return (100 - 51) / (154 - 55) * (x - 55) + 51
    elif x <= 254:
        return (101 - 150) / (254 - 155) * (x - 155) + 101
    elif x <= 354:
        return (151 - 200) / (354 - 255) * (x - 255) + 151
    elif x <= 424:
        return (300 - 201) / (424 - 355) * (x - 355) + 201
    elif x <= 504:
        return (400 - 301) / (504 - 425) * (x - 425) + 301
    elif x > 504:
        return (500 - 401) / (604 - 505) * (x - 505) + 401 
    else:
        return 0

## SO2 Sub-Index calculation
def get_so2_aqi(x, hours):
    if hours == 1:
        if x <= 35.0:
            return (50 - 0) / (35 - 0) * (x - 0) + 0
        elif x <= 75:
            return (100 - 51) / (75 - 36) * (x - 36) + 51
        elif x <= 185:
            return (101 - 150) / (185 - 76) * (x - 76) + 101
        elif x <= 304:
            return (151 - 200) / (304 - 186) * (x - 186) + 151
        else: 
            return 0
    if hours == 24:
        if x <= 604:
            return (300 - 201) / (604 - 305) * (x - 305) + 201
        elif x <= 804:
            return (400 - 301) / (804 - 605) * (x - 605) + 301
        elif x > 804:
            return (500 - 401) / (1004 - 805) * (x - 805) + 401 
        else:
            return 0
    
def get_no2_aqi(x):
    if x <= 53.0:
        return (50 - 0) / (53 - 0) * (x - 0) + 0
    elif x <= 100:
        return (100 - 51) / (100 - 54) * (x - 54) + 51
    elif x <= 360:
        return (101 - 150) / (360 - 101) * (x - 101) + 101
    elif x <= 649:
        return (151 - 200) / (649 - 361) * (x - 361) + 151
    elif x <= 1249:
        return (300 - 201) / (1249 - 650) * (x - 650) + 201
    elif x <= 1649:
        return (400 - 301) / (1649 - 1250) * (x - 1250) + 301
    elif x > 1649:
        return (500 - 401) / (2049 - 1650) * (x - 1650) + 401 
    else:
        return 0
    
# def get_NH3_subindex(x):
#     if x <= 200:
#         return x * 50 / 200
#     elif x <= 400:
#         return 50 + (x - 200) * 50 / 200
#     elif x <= 800:
#         return 100 + (x - 400) * 100 / 400
#     elif x <= 1200:
#         return 200 + (x - 800) * 100 / 400
#     elif x <= 1800:
#         return 300 + (x - 1200) * 100 / 600
#     elif x > 1800:
#         return 400 + (x - 1800) * 100 / 600
#     else:
#         return 0
    
def get_co_aqi(x): #default hours is 8
    x = x / 1000 #ppb to ppm
    if x <= 4.4:
        return (50 - 0) / (4.4 - 0) * (x - 0) + 0
    elif x <= 9.4:
        return (100 - 51) / (9.4 - 4.5) * (x - 4.5) + 51
    elif x <= 12.4:
        return (101 - 150) / (12.4 - 9.5) * (x - 9.5) + 101
    elif x <= 15.4:
        return (151 - 200) / (15.4 - 12.5) * (x - 12.5) + 151
    elif x <= 30.4:
        return (300 - 201) / (30.4 - 15.5) * (x - 15.5) + 201
    elif x <= 40.4:
        return (400 - 301) / (40.4 - 30.5) * (x - 30.5) + 301
    elif x > 40.4:
        return (500 - 401) / (50.4 - 40.5) * (x - 40.5) + 401 
    else:
        return 0
    
def get_o3_aqi(x, hours):
    if hours == 8:
        if x <= 54: #8hr
            return (50 - 0) / (54 - 0) * (x - 0) + 0
        elif x <= 70:
            return (100 - 51) / (70 - 55) * (x - 55) + 51
        elif x <= 85:
            return (150 - 101) / (85 - 71) * (x - 71) + 101
        elif x <= 105:
            return (200 - 151) / (105 - 86) * (x - 86) + 151
        elif x <= 200:
            return (300 - 201) / (200 - 106) * (x - 106) + 201
        else:
            return 0
    if hours == 1:
        if x <= 164:
            return (150 - 101) / (164 - 125) * (x - 125) + 101
        if x <= 204: #1hr 
            return (200 - 151) / (204 - 165) * (x - 165) + 151
        elif x <= 404:
            return (300 - 201) / (404 - 205) * (x - 205) + 201
        elif x <= 504:
            return (400 - 301) / (504 - 405) * (x - 405) + 301
        elif x > 504:
            return (500 - 401) / (604 - 505) * (x - 505) + 401
        else:
            return 0
    return 0
def tavg(lat, lon, start, hours):
    
    # Format datetime object as YYYY-MM-DD
    if start%3600 != 0:
        if abs(3600 - start%3600) > start%3600:
            start -= start%3600
        else:
            start += 3600 - start%3600
    
    sdate = datetime.datetime.fromtimestamp(start - 24 * 60 * 60).strftime(f'%Y-%m-%d')#T%H:%M')
    edate =  datetime.datetime.fromtimestamp(start + 24 * 60 * 60).strftime(f'%Y-%m-%d')
    sf = datetime.datetime.fromtimestamp(start).strftime(f'%Y-%m-%dT%H:%M')
    # print(sdate, edate)
    #print(sf)
    tdata = requests.get(f'https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={sdate}&end_date={edate}&hourly=temperature_2m').json()
    #print(tdata)
    times = tdata['hourly'].get('time')
    #print(times)
    p = times.index(sf)
    #print("got hre")
    times = times[p-hours+1:p+1]
    temp = tdata['hourly'].get('temperature_2m')
    temp = temp[p-hours+1:p+1]
    print(temp)
    
    avg = sum(temp) / len(temp)
    #print("Temperature:", avg)
    return avg

def get_avg_ppb(pollutant, hours, data, s):
    #print(f"getting ppb of {pollutant}")
   
    lat = s[0]
    lon = s[1]
    t = s[2]
    average_t = tavg(lat, lon, t, hours)
    # if pollutant == 'pm10':
    #     print("PM10 data", len(data))
    data = data[-1*hours:]#data[len(data)-1-hours:len(data)-1]
    # if pollutant == 'pm10':
    #     print("PM10 data", data)
    total_p = 0
    #total_t = 0
    for i in data:
        #print("Val:", i['components'].get(pollutant))
        total_p += i['components'].get(pollutant)

    average_p = total_p/hours
    #print("Average Pollutant val", average_p)

    molecular_weights = {
        'co': 28.01,
        'no2': 46.01,
        'o3': 48.00,
        "so2": 64.07,
    }
    if pollutant != 'pm10' and pollutant != 'pm2_5':
        ppb = average_p / (12.187 * molecular_weights.get(pollutant)) * (273.15 + average_t)
    else:
        ppb = average_p
    #print(f"PPB of {pollutant}", ppb)
    ppbs[pollutant] = ppb
    #print("PPBDICT", ppbs)
    return [ppb]



def get_aqi(s):#get_aqi(lat, lon, time):
    
    data = list(map(float, s.replace(".png", "").split("_"))) #formatting the file name
    lat = data[1]
    lon = data[0]
    time = int(data[2] / 1000)
    window = 24 * 60 * 60 #* 1000 #24 hour window in XmilliX  seconds??
    start = time - window + 3600
    end = time + 3600
    #cnt = 24
    #(start, end)
    response = requests.get(f'http://pro.openweathermap.org/data/2.5/air_pollution/history?lat={lat}&lon={lon}&start={start}&end={end}&appid={api_key}')
    full_data = response.json()
    
    
    try:
        #tdata = temp_data['place']['holders'] #select past 24 hours of data
        
        aqi_data = full_data['list']#[0] #components are in ppb (parts per billion)
        #print(aqi_data)
        # for i in aqi_data:
        #     print(i['components'].get('o3'))
        d = compute_aqi(aqi_data, [lat, lon, time]) + [lat, lon, time]
        print(d)
        d.insert(0, s)
        return(d)
    except:
        pass

def compute_aqi(pollutants, s): #in ug/m3
    
    #print("computing")
    co_aqi = get_co_aqi(get_avg_ppb('co', 8, pollutants, s))
    # print("COAQI", co_aqi)
    # print("--------------------")
    no2_aqi = get_no2_aqi(get_avg_ppb('no2', 1, pollutants, s))
    # print("no2AQI", no2_aqi)
    # print("--------------------")
    pm10_aqi = get_pm10_aqi(get_avg_ppb('pm10', 24, pollutants, s))
    # print("pm10AQI", pm10_aqi)
    # print("--------------------")
    pm2_5_aqi = get_pm10_aqi(get_avg_ppb('pm2_5', 24, pollutants, s))
    # print("pm2_5AQI", pm2_5_aqi)
    # print("--------------------")
    o3_aqi = get_o3_aqi(get_avg_ppb('o3', 8, pollutants, s), 8)
    # print("o3AQI", o3_aqi)
    # print("--------------------")
    if o3_aqi > 300:
        o3_aqi = get_o3_aqi(get_avg_ppb('o3', 1, pollutants, s), 1)
    so2_aqi = get_so2_aqi(get_avg_ppb('so2', 1, pollutants, s), 1)
    # print("so2AQI", so2_aqi)
    # print("--------------------")
    if so2_aqi > 300:
        so2_aqi = get_so2_aqi(get_avg_ppb('so2', 24, pollutants, s), 24)

    vals = {
        co_aqi: "co",
        no2_aqi: "no2",
        o3_aqi: "o3",
        so2_aqi: "so2",
        pm2_5_aqi: "pm2_5",
        pm10_aqi: "pm10",
    }
    #print("hi")
    
    max_aqi = max(vals.keys())
    #print(max_aqi)
    lead_pollutant = vals.get(max_aqi)
    #print(lead_pollutant)
    #lead_amt = list(vals.keys())[list(vals.values()).index(lead_pollutant)]
    #print(ppbs)
    lead_amt = ppbs.get(lead_pollutant)
    #print(lead_amt)
    
    return([max_aqi, lead_pollutant, lead_amt])

#[-104.67045045045049, 39.65945945945949, 1627430400000.0]
#get_aqi(39.6594, -104.6704, 1627430400000 / 1000)
#get_aqi(34, 118.233, 1610000000)

# print(get_aqi("-70.9884684684685_42.29945945945949_1629936000000.png"))

print(get_aqi("-87.65_41.85_1649728000000.png"))

        
