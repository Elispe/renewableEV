import numpy as np
import sys

case_name = sys.argv[1]

# Enter number of iterations
num_iter = 100

numReq = np.load('numReq.npy')
h_format = np.load('h_format.npy')
y_power_cars = np.load('y_power_cars1.npy')
miss_ride_time = np.load('miss_ride_time1.npy')
high_battery_time = np.load('high_battery_time1.npy')
int_battery_time = np.load('int_battery_time1.npy')
low_battery_time = np.load('low_battery_time1.npy')
ev_ride_time = np.load('ev_ride_time1.npy')
ev_charge_time = np.load('ev_charge_time1.npy')
ev_idle_time = np.load('ev_idle_time1.npy')
if case_name == "rideSharing":
    y_Pref_current = np.load('y_Pref_current.npy')
    incent_charge_assigned = np.load('incent_charge_assigned1.npy')
    incent_ride_assigned = np.load('incent_ride_assigned1.npy')

for i in range(2, num_iter + 1):
    y_power_cars += np.load('y_power_cars' + str(i) + '.npy')
    miss_ride_time += np.load('miss_ride_time' + str(i) + '.npy')
    high_battery_time += np.load('high_battery_time' + str(i) + '.npy')
    int_battery_time += np.load('int_battery_time' + str(i) + '.npy')
    low_battery_time += np.load('low_battery_time' + str(i) + '.npy')
    ev_ride_time += np.load('ev_ride_time' + str(i) + '.npy')
    ev_charge_time += np.load('ev_charge_time' + str(i) + '.npy')
    ev_idle_time += np.load('ev_idle_time' + str(i) + '.npy')
    if case_name == "rideSharing":
        incent_charge_assigned += np.load('incent_charge_assigned' + str(i) + '.npy')
        incent_ride_assigned += np.load('incent_ride_assigned' + str(i) + '.npy')

# Average data, not rounded
y_power_cars = y_power_cars / num_iter
miss_ride_time = miss_ride_time / num_iter
if case_name == "rideSharing":
    incent_charge_assigned = incent_charge_assigned / num_iter
    incent_ride_assigned = incent_ride_assigned / num_iter
# Average data, rounded
high_battery_time = np.rint(high_battery_time / num_iter)
int_battery_time = np.rint(int_battery_time / num_iter)
low_battery_time = np.rint(low_battery_time / num_iter)
ev_ride_time = np.rint(ev_ride_time / num_iter)
ev_charge_time = np.rint(ev_charge_time / num_iter)
ev_idle_time = np.rint(ev_idle_time / num_iter)

# Avg results
time_slot = 15  # 15-min time slots

print("  *** Avg results *** ")
print("  --- Vehicles with low battery: " + str(low_battery_time[-1]))
print("  --- Vehicles with int battery: " + str(int_battery_time[-1]))
print("  --- Vehicles with high battery: " + str(high_battery_time[-1]))
print("Missed ride-req, sum min by min: " + str(sum(miss_ride_time)))
print("Missed ride-req, rounded 5-min sum: " + str(
    int(sum(np.rint(np.sum(np.array(miss_ride_time).reshape(len(miss_ride_time) // time_slot, time_slot), axis=1))))))
print("QoS: " + str(100 - (int(sum(np.rint(
    np.sum(np.array(miss_ride_time).reshape(len(miss_ride_time) // time_slot, time_slot), axis=1))))) / numReq * 100))

if case_name == "rideSharing":
    lost_power_percent = 0
    for i in range(len(y_Pref_current)):
        if sum(y_Pref_current[i]) > sum(y_power_cars[i]):
            lost_power_percent += (sum(y_Pref_current[i]) - sum(y_power_cars[i]))

    tot_Pref = sum([sum(i) for i in y_Pref_current])
    print("Power lost: " + str(round((lost_power_percent / tot_Pref) * 100, 2)) + "%")

# Save results for later
path = ''
np.save(path + 'miss_ride_time', miss_ride_time)
np.save(path + 'y_power_cars', y_power_cars)
np.save(path + 'high_battery_time', high_battery_time)
np.save(path + 'int_battery_time', int_battery_time)
np.save(path + 'low_battery_time', low_battery_time)
np.save(path + 'ev_ride_time', ev_ride_time)
np.save(path + 'ev_charge_time', ev_charge_time)
np.save(path + 'ev_idle_time', ev_idle_time)
if case_name == "rideSharing":
    np.save(path + 'incent_charge_assigned', incent_charge_assigned)
    np.save(path + 'incent_ride_assigned', incent_ride_assigned)
