"""
Extract ride request data from the Taxi and Limousine Commission, Manhattan, NY
Choose the size of a subsample for the time window of interest
"""
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import datetime

np.random.seed(8)

#Import yellow taxi data
trips = pq.read_table('yellow_tripdata_2022-03.parquet')
trips = trips.to_pandas()

#Choose which days you want the data for
mask = (trips['tpep_pickup_datetime'] >= '2022-03-01') & (trips['tpep_pickup_datetime'] < '2022-03-02')
oneDayTrips = trips.loc[mask]
selectedTrips = oneDayTrips.loc[:, ['tpep_pickup_datetime', 'PULocationID', 'DOLocationID']]
orderedTrips = selectedTrips.sort_values(by="tpep_pickup_datetime")

#Import locationID map
df = pd.read_csv('taxiZones.csv')
df = df.loc[:, ['LocationID', 'Borough', 'Zone']]
ManhattanLoc = df.loc[df['Borough'] == 'Manhattan']
ManhattanLoc = ManhattanLoc.loc[:, ['LocationID', 'Zone']]

#Group zones into 18 areas
darea = {'LocationID': [12, 13, 261, 87, 88, 209, 125, 211, 144, 231, 45, 148, 232, 158, 249, 113, 114, 79, 4, 68, 90, 234, 107, 224, 246, 100, 186, 164, 170, 137, 50, 48, 230, 163, 161, 162, 229, 233, 143, 142, 239, 141, 140, 237, 24, 151, 238, 236, 263, 262, 166, 41, 74, 75, 42, 152, 116], 'Area': [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6, 6, 6, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 13, 13, 13, 14, 14, 14, 15, 15, 15, 16, 16, 17, 17, 18, 18, 18]}
dfArea = pd.DataFrame(data=darea)
areasTable = pd.merge(ManhattanLoc, dfArea, how='left', left_on=['LocationID'], right_on=['LocationID'])
areas = areasTable.loc[:, ['LocationID', 'Area']]
final = pd.merge(orderedTrips, areas, how='left', left_on=['PULocationID'], right_on=['LocationID'])
final = final.drop('LocationID', axis=1)
final = pd.merge(final, areas, how='left', left_on=['DOLocationID'], right_on=['LocationID'])
final = final.drop('LocationID', axis=1)
final = final.rename(columns={"Area_x": "PUArea", "Area_y": "DOArea"})
maskNaN = (final['PUArea'] > 0) & (final['DOArea'] > 0)
final = final.loc[maskNaN]
final['PUArea'] = final['PUArea'].astype('Int64')
final['DOArea'] = final['DOArea'].astype('Int64')

#Keep only trips within the 9 areas in lower Manhattan
mask9 = (final['PUArea'] <= 9) & (final['DOArea'] <= 9)
final9 = final.loc[mask9]

#Keep only ride requests received within the time window specified below
h_in = 6
ini = datetime.datetime(2022, 3, 1, h_in, 0, 0)
one_min = datetime.timedelta(seconds=60)
num_min = 18*60 # 18 hours

records = {}
numRequestsRed = 0

for i in range(num_min):
    mask9min = (final9['tpep_pickup_datetime'] >= ini) & (final9['tpep_pickup_datetime'] < ini + one_min)
    final9min = final9.loc[mask9min]
    numRequests = len(final9min)
    PU_arr = final9min['PUArea'].tolist()
    DO_arr = final9min['DOArea'].tolist()

    # print(str(numRequests) + " requests between " + str(ini) + " and " + str(ini + one_min))
    scalingFactor = 8
    # print("Keep: " + str(round(numRequests / scalingFactor)))
    numRequestsRed += round(numRequests / scalingFactor)

    # Scale down the number of requests/minute by scalingFactor
    index = []
    while len(index) < round(numRequests / scalingFactor):
        rand_num = np.random.randint(0, numRequests)
        if rand_num not in index:
            index.append(rand_num)
    index.sort()

    PUReduced = []
    DOReduced = []
    for j in index:
        PUReduced.append(PU_arr[j])
        DOReduced.append(DO_arr[j])
    records[i] = [PUReduced, DOReduced]

    ini = ini + one_min

print("Total number riding requests: " + str(numRequestsRed))

#Import PV profiles
df_pvprofile = pd.read_csv('PV_norm.csv')
pv_sunny = df_pvprofile['Sunny']
pv_cloud_am = df_pvprofile['Cloud am']
pv_cloud_pm = df_pvprofile['Cloud pm']

# Parameters to tune
c_RES_total = np.array([])
c_RES_aux = [0.00001, 0.0001, 0.001, 0.1, 0.1, 0.001, 0.0005, 0.0001, 0.00001, 0.000001]
h_aux = np.array([])
h_aux_bid = [0.00001, 0.001, 0.1, 0.5, 0.75, 0.75, 0.5, 0.05, 0.00001, 0.000001]
h_aux_bid_min = [0, 0, 0.09, 0.35, 0.55, 0.55, 0.35, 0.04, 0, 0]
b_aux_min = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
alpha_w_total = np.array([])
alpha_w_aux = [0, 0, 0.01, 0.05, 0.1, 0.1, 0.1, 0.05, 0, 0]
beta_w_total = np.array([])
beta_w_aux = [0, 0.1, 0.1, 0.1, 0.08, 0.05, 0.01, 0, 0, 0]
h_aux_min = np.array([])
b_aux = np.array([])
for i in range(len(c_RES_aux)):
    c_RES_total_aux = 1*c_RES_aux[i]*np.ones(num_min // (len(c_RES_aux)))
    c_RES_total = np.concatenate([c_RES_total, c_RES_total_aux])
    alpha_w_total_aux = 1*alpha_w_aux[i]*np.ones(num_min // (len(alpha_w_aux)))
    alpha_w_total = np.concatenate([alpha_w_total, alpha_w_total_aux])
    beta_w_total_aux = 1*beta_w_aux[i]*np.ones(num_min // (len(beta_w_aux)))
    beta_w_total = np.concatenate([beta_w_total, beta_w_total_aux])
    h_aux1 = 1*h_aux_bid[i]*np.ones(num_min // (len(h_aux_bid)))
    h_aux = np.concatenate([h_aux, h_aux1])
    h_aux1_min = 1*h_aux_bid_min[i]*np.ones(num_min // (len(h_aux_bid_min)))
    h_aux_min = np.concatenate([h_aux_min, h_aux1_min])
    b_aux1 = 1*b_aux_min[i]*np.ones(num_min // (len(b_aux_min)))
    b_aux = np.concatenate([b_aux, b_aux1])