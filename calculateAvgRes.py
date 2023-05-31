import numpy as np
import matplotlib.pyplot as plt
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

# Plots
# Pref and missing rides

# Uncomment below for latex
# plt.rcParams.update({"text.usetex": True, "font.family": "lmodern", "font.size": 12})
plt.rcParams['axes.xmargin'] = 0

fig, host = plt.subplots(figsize=(5, 4), layout="constrained")  # layout='constrained')  # (width, height) in inches
ax3 = host.twinx()
host.set_xlabel("Time")
host.set_ylabel("Power [kW]")
ax3.set_ylabel("Missed ride requests")
ax3.locator_params(axis="y", integer=True, tight=True)
if case_name == 'rideSharing':
    p1 = host.plot(h_format, [sum(i) for i in y_Pref_current], label='$P_{\mathrm{ref}}$', color="#1EBFE1")
    p1b = host.plot(h_format, [sum(i) for i in y_power_cars], label='$v_{\mathrm{ch}} p_{\mathrm{ch}}$',
                    color="#1E5EE1")
    host.legend(handles=p1 + p1b, loc='upper right', fontsize="14", ncol=1)
else:
    p1b = host.plot(h_format, y_power_cars, label='$v_{\mathrm{ch}} p_{\mathrm{ch}}$', color="#1E5EE1")
    host.legend(handles=p1b, loc='upper left', fontsize="12", ncol=1)
p3 = ax3.bar(h_format[::time_slot],
             np.rint(np.sum(np.array(miss_ride_time).reshape(len(miss_ride_time) // time_slot, time_slot), axis=1)),
             width=7, alpha=0.3, color="#C8377E", label=("Total missed ride requests: " + str(int(sum(
        np.rint(np.sum(np.array(miss_ride_time).reshape(len(miss_ride_time) // time_slot, time_slot), axis=1)))))))
ax3.legend(loc='upper center', fontsize="14", ncol=1, bbox_to_anchor=(0.5, 1.15))
host.set_xticks(h_format[::360])
host.set_xticklabels(h_format[::360])
plt.savefig(case_name + "_MissRide_avg.pdf", bbox_inches='tight')

# SOC

# plt.rcParams.update({"text.usetex": True, "font.family": "lmodern", "font.size": 12})
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
fig3.savefig(case_name + '_SOC_avg.pdf', bbox_inches='tight')

# EV availability

# plt.rcParams.update({"text.usetex": True, "font.family": "lmodern", "font.size": 12})
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
fig7.savefig(case_name + '_EV_avg.pdf', bbox_inches='tight')

if case_name == 'rideSharing':
    # Pref and charging profile at each station

    # plt.rcParams.update({"text.usetex": True, "font.family": "lmodern", "font.size": 7.5})

    plt.rcParams['axes.xmargin'] = 0

    fig4, axs = plt.subplots(1, 4, figsize=(8, 1.5), tight_layout=True)
    for i in range(4):
        plt.rcParams['axes.xmargin'] = 0
        axs[i].plot(h_format, [y_Pref_current[k][i] for k in range(len(y_Pref_current))], label='$P_{\mathrm{ref}}$',
                    color="#1EBFE1")
        axs[i].plot(h_format, [y_power_cars[k][i] for k in range(len(y_power_cars))],
                    label='$v_{\mathrm{ch}} p_{\mathrm{ch}}$', color="#1E5EE1")
        axs[i].set_xticks(h_format[::360])
        axs[i].set_xticklabels(h_format[::360])
        axs[i].set_xlabel('Time')
    axs[0].set_ylabel('Power [kW]')
    axs[3].legend(loc='upper right', ncol=1)
    fig4.savefig(case_name + '_power4stations_avg.pdf', bbox_inches='tight')

    # Incentives assigned

    # plt.rcParams.update({"text.usetex": True, "font.family": "lmodern", "font.size": 11})

    plt.rcParams['axes.xmargin'] = 0

    fig5, ax5 = plt.subplots(figsize=(4, 2), tight_layout=True)
    ax5.bar(h_format[::time_slot],
            np.sum(np.array(incent_charge_assigned).reshape(len(incent_charge_assigned) // time_slot, time_slot),
                   axis=1), width=0.5, alpha=0.5, label="Charge request", color="#33397E")
    ax5.bar(h_format[::time_slot],
            np.sum(np.array(incent_ride_assigned).reshape(len(incent_ride_assigned) // time_slot, time_slot), axis=1),
            width=0.5, alpha=0.5, label="Ride request", color="#7e7833")
    ax5.set_xticks(h_format[::360])
    ax5.set_xticklabels(h_format[::360])
    ax5.set_xlabel('Time')
    ax5.set_ylabel('Incentive[$\$$]')
    ax5.margins(y=0)
    ax5.set_ylim(-1, 11)
    ax5.legend(loc='upper center', fontsize="10", ncol=2)
    fig5.savefig(case_name + '_incAssign_avg.pdf', bbox_inches='tight')
