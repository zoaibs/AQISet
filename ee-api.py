import ee
import urllib.request


#AUTHENTICATION ON STARTUP

# ee.Authenticate()
ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')


#Main Code
import csv
dataset = ee.ImageCollection('USDA/NAIP/DOQQ').filter(ee.Filter.date('2021-01-01', '2023-06-01')).select(['R', 'G', 'B'])
size = 1000000 #in meters, so 10km^2 per city
urls = []
fields = []
rows = []


# reading csv file
with open("coords.csv", 'r') as csvfile:
    # creating a csv reader object
    csvreader = csv.reader(csvfile)

    # extracting field names through first row
    fields = next(csvreader)

    # extracting each data row one by one
    for row in csvreader:
        rows.append(row)
# print(len(rows))
# print(rows)

imgs = []
coordsList = rows

from multiprocessing import Pool

import numpy as np

km_per_deg = 1/111 #km per degree longitude or latitude

def get_url(data):
    # city = data[0]
    # state = data[1]
    center_y = float(data[2])
    center_x = float(data[3])

    for i in np.arange(center_y + 4.5 * km_per_deg, center_y - 5.5 * km_per_deg, -1 * km_per_deg): #only a 9x9
        for j in np.arange(center_x - 4.5 * km_per_deg, center_x + 5.5 * km_per_deg, km_per_deg):
    # for i in np.arange(center_y + 1/111, center_y - 2/111, -1*1/111):
    #     for j in np.arange(center_x - 1/111, center_x + 2/111, 1*1/111):   
          print(j,i)   
          point = ee.Geometry.Point([j, i])
          bound_box = point.buffer(ee.Number(size).sqrt().divide(2), 1).bounds()
          intersect = dataset.filterBounds(bound_box).mosaic()
          try:
              url = intersect.getThumbURL({
                'region': bound_box,
                'dimensions': '1000x1000',
                'format': 'png'})
              
              time = str(dataset.filterBounds(bound_box).first().date().millis().getInfo())
              # print(time)
              # print(type(time))
              # r = requests.get(url, stream=True)
              # if r.status_code != 200:
              #     raise r.raise_for_status()
              filename = "sample_img/" + str(j) + "_" + str(i) + "_" + str(time) + ".png"
              # print(filename)
              
              urllib.request.urlretrieve(url, filename)
              # with open(filename, 'wb') as out_file:
              #     shutil.copyfileobj(r.raw, out_file)
              #print("Done: ", index)
          except:
            #wtf
              pass


if __name__ == "__main__":
    count = 0
    pool = Pool(25)
    pool.map(get_url, coordsList)
    pool.close()
