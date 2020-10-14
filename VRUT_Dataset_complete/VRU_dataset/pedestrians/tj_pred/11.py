import pandas as pd
import random
LON1 = 121.4135
LON2 = 121.4176
LAT1 = 31.2165
LAT2 = 31.3233
lon =[]
lat =[]
for i in range(100):
    lon.append(round(random.uniform(LON1, LON2), 4))
    lat.append(round(random.uniform(LAT1, LAT2), 4))
c={"lon":lon,
   'lat':lat}
data= pd.DataFrame(c)
def generalID(lon,lat,column_num,row_num):
    # 若在范围外的点，返回-1
    if lon <= LON1 or lon >= LON2 or lat <= LAT1 or lat >= LAT2:
        return -1
    # 把经度范围根据列数等分切割
    column = (LON2 - LON1)/column_num
    # 把纬度范围根据行数数等分切割
    row = (LAT2 - LAT1)/row_num
    # 二维矩阵坐标索引转换为一维ID，即： （列坐标区域（向下取整）+ 1） + （行坐标区域 * 列数）
    return int((lon-LON1)/column)+ 1 + int((lat-LAT1)/row) * column_num
data['label'] = data.apply(lambda x: generalID(x['lon'], x['lat'],4,4), axis = 1)
print(data.head(10))
