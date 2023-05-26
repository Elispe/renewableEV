import numpy as np
import matplotlib.pyplot as plt

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

for i in range(2, num_iter + 1):
    y_power_cars += np.load('y_power_cars' + str(i) + '.npy')
    miss_ride_time += np.load('miss_ride_time' + str(i) + '.npy')
    high_battery_time += np.load('high_battery_time' + str(i) + '.npy')
    int_battery_time += np.load('int_battery_time' + str(i) + '.npy')
    low_battery_time += np.load('low_battery_time' + str(i) + '.npy')
    ev_ride_time += np.load('ev_ride_time' + str(i) + '.npy')
    ev_charge_time += np.load('ev_charge_time' + str(i) + '.npy')
    ev_idle_time += np.load('ev_idle_time' + str(i) + '.npy')

# Average data, not rounded
y_power_cars = y_power_cars / num_iter
miss_ride_time = miss_ride_time / num_iter
# Average data, rounded
high_battery_time = np.rint(high_battery_time / num_iter)
int_battery_time = np.rint(int_battery_time / num_iter)
low_battery_time = np.rint(low_battery_time / num_iter)
ev_ride_time = np.rint(ev_ride_time / num_iter)
ev_charge_time = np.rint(ev_charge_time / num_iter)
ev_idle_time = np.rint(ev_idle_time / num_iter)

# Avg results
time_slot = 15  # 15-min time slots

print("  *** Avg results, rounded *** ")
print("  --- Vehicles with low battery: " + str(low_battery_time[-1]))
print("  --- Vehicles with int battery: " + str(int_battery_time[-1]))
print("  --- Vehicles with high battery: " + str(high_battery_time[-1]))
print("Missed ride-req, sum min by min: " + str(sum(miss_ride_time)))
print("Missed ride-req, rounded 5-min sum: " + str(
    int(sum(np.rint(np.sum(np.array(miss_ride_time).reshape(len(miss_ride_time) // time_slot, time_slot), axis=1))))))
print("QoS: " + str(100 - (int(sum(np.rint(
    np.sum(np.array(miss_ride_time).reshape(len(miss_ride_time) // time_slot, time_slot), axis=1))))) / numReq * 100))

# Plots
# Pref and missing rides
plt.rcParams.update({"text.usetex": True, "font.family": "lmodern", "font.size": 12})
plt.rcParams['axes.xmargin'] = 0

fig, host = plt.subplots(figsize=(4.2, 2.55),
                         layout="constrained")  # layout='constrained')  # (width, height) in inches
ax3 = host.twinx()
host.set_xlabel("Time")
host.set_ylabel("Power [kW]")
ax3.set_ylabel("Missed ride requests")
ax3.locator_params(axis="y", integer=True, tight=True)
p1b = host.plot(h_format, y_power_cars, label='$v_{\mathrm{ch}} p_{\mathrm{ch}}$', color="#1E5EE1")
p3 = ax3.bar(h_format[::time_slot],
             np.rint(np.sum(np.array(miss_ride_time).reshape(len(miss_ride_time) // time_slot, time_slot), axis=1)),
             width=7, alpha=0.3, color="#C8377E", label=("Total missed ride requests: " + str(int(sum(
        np.rint(np.sum(np.array(miss_ride_time).reshape(len(miss_ride_time) // time_slot, time_slot), axis=1)))))))
ax3.legend(loc='upper center', fontsize="11", ncol=1, bbox_to_anchor=(0.48, 1.25))
host.legend(handles=p1b, loc='upper left', fontsize="12", ncol=1)
host.set_xticks(h_format[::360])
host.set_xticklabels(h_format[::360])
plt.savefig("NoChargeReq_MissRide_avg.pdf", bbox_inches='tight')

# SOC
plt.rcParams.update({"text.usetex": True, "font.family": "lmodern", "font.size": 12})
plt.rcParams['axes.xmargin'] = 0

fig3, ax3 = plt.subplots(figsize=(4.3, 3), tight_layout=True)
ax3.bar(h_format, high_battery_time, width=1.0, alpha=0.5, label="High SOC", color="limegreen")
ax3.bar(h_format, int_battery_time, width=1.0, alpha=0.5, label="Mid SOC", color="gold")
ax3.bar(h_format, low_battery_time, width=1.0, alpha=0.6, label="Low SOC", color="orangered")
ax3.set_xticks(h_format[::360])
ax3.set_xticklabels(h_format[::360])
ax3.set_xlabel('Time')
ax3.set_ylabel('Number of EV')
ax3.margins(y=0)
ax3.legend(loc='upper center', fontsize="11", bbox_to_anchor=(0.48, 1.25), ncol=3)
fig3.savefig('NoChargeReq_SOC_avg.pdf', bbox_inches='tight')

# EV availability
plt.rcParams.update({"text.usetex": True, "font.family": "lmodern", "font.size": 12})
plt.rcParams['axes.xmargin'] = 0

fig7, ax7 = plt.subplots(figsize=(4.2, 3), tight_layout=True)
ax7.bar(h_format, ev_idle_time, width=1.0, alpha=0.6, label="Idling", color="#C53AA6")
ax7.bar(h_format, ev_ride_time, width=1.0, alpha=0.5, label="Riding", color="#3AA6C5")
ax7.bar(h_format, ev_charge_time, width=1.0, alpha=0.5, label="Charging", color="#A6C53A")
ax7.set_xticks(h_format[::360])
ax7.set_xticklabels(h_format[::360])
ax7.set_xlabel('Time')
ax7.set_ylabel('Number of EV')
ax7.margins(y=0)
ax7.legend(loc='upper center', fontsize="11", bbox_to_anchor=(0.48, 1.25), ncol=3)
fig7.savefig('NoChargeReq_EV_avg.pdf', bbox_inches='tight')
