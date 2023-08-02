# This is the main Python script for the business-as-usual case.
# No charge-request are sent. EVs charge when their battery is below min_charge.

import numpy as np
import networkx as nx
import pulp
import tripData as tData
import time
import datetime
import sys

from Vehicle import Vehicle
from Request import RideRequest

# Seed corresponds to SLURM_ARRAY_TASK_ID
seed = int(sys.argv[1])
np.random.seed(seed)

# Fleet size
n_veh = 100

# Variable initialization
delta_ride = 2  # Max zones away from customer for pick-up
power_transferred = 12  # kW for each car charging
discharge_rate = 0.1  # kWh/minute
charge_rate = discharge_rate * 2
min_travel_time = 5  # min
min_consume = min_travel_time * discharge_rate
infeasib_cost = 1e5
infeasib_threas = 1e4
travel_edge_time = 10
in_value = 1.0 / n_veh  # for initialization of x_ij

# Generate random soc array
random_soc = [np.random.uniform(Vehicle.min_charge, Vehicle.full_charge) for n in range(n_veh)]

# DiGraph - unweighted
g = nx.DiGraph()
elist = [(1, 2), (1, 3), (2, 1), (2, 3), (2, 4), (2, 5), (3, 1), (3, 2), (3, 4), (3, 5), (4, 2), (4, 3), (4, 5),
         (4, 6),
         (4, 7), (5, 2), (5, 3), (5, 4),
         (5, 6), (5, 7), (6, 4), (6, 5), (6, 7), (6, 8), (6, 9), (7, 4), (7, 6), (7, 8), (7, 9), (8, 6), (8, 7),
         (8, 9),
         (9, 6), (9, 7), (9, 8)]
g.add_edges_from(elist)
numNodes = g.number_of_nodes()

# Initialize vehicle positions and soc.
np.random.seed(0)
vehicles = []
for j in range(n_veh // numNodes):
    vehicles.extend([Vehicle(pos, Vehicle.full_charge) for pos in range(1, numNodes + 1)])
for j in range(n_veh - len(vehicles)):
    vehicles.append(Vehicle(np.random.randint(1, numNodes + 1), Vehicle.full_charge))

# Randomize the initial soc, otherwise comment below to get fully charged EV
for j, veh in enumerate(vehicles):
    veh.set_soc(random_soc[j])

# Track charging
y_power_cars = []

# Variables to track performance
miss_ride_time = []
low_battery_time = []
int_battery_time = []
high_battery_time = []
ev_ride_time = []
ev_charge_time = []
ev_idle_time = []
h_format = []

# Iterate over time, deltaT = 1 min
for k in range(tData.num_min):
    minute = k + tData.h_in * 60
    PULoc = tData.records[k][0]
    DOLoc = tData.records[k][1]
    num_ride_req = len(PULoc)
    h_format.append(time.strftime("%H:%M", time.gmtime(minute * 60)))
    # to run over multiple days use code below instead
    # dt = datetime.datetime(2022, 3, 1) - datetime.datetime(1970, 1, 1)
    # minutessince = int(dt.total_seconds() / 60)
    # h_format.append(time.strftime("%b %d %H:%M", time.gmtime((minutessince + minute) * 60)))

    print("*** Minute: " + str(k) + " ***")

    req_vec = []  # List of requests collected between time t and t+1
    req_idx = 0

    start_costs = [0] * n_veh  # costs to get to the pick-up point
    start_path = [0] * n_veh  # path to get to the pick-up point

    cost = []
    paths = []

    # Update vehicle positions
    for veh in vehicles:
        # If vehicle reaches passenger final destination
        if not veh.is_available() and isinstance(veh.request, RideRequest) and veh.get_estimated_arrival() <= k:
            veh.set_position(veh.get_request().get_destination())
            veh.terminate_request()  # Vehicle again available
        # If vehicle runs out of battery ---> must charge, becomes unavailable
        if veh.is_available() and not veh.is_charging() and veh.get_soc() < Vehicle.min_charge:
            veh.charge()

    # Update vehicle soc
    # If vehicle is fully charged, disconnect
    for veh in vehicles:
        if veh.get_estimated_arrival() >= k:
            veh.discharge(discharge_rate)
        elif veh.is_charging():
            veh.charge(charge_rate)

    # Track soc status
    low_soc_count = 0
    int_soc_count = 0
    high_soc_count = 0
    for veh in vehicles:
        if veh.get_soc() < Vehicle.min_charge:
            low_soc_count += 1
        elif veh.get_soc() < Vehicle.int_charge:
            int_soc_count += 1
        else:
            high_soc_count += 1

    low_battery_time.append(low_soc_count)
    int_battery_time.append(int_soc_count)
    high_battery_time.append(high_soc_count)

    # Track EV availability
    riding_ev_count = 0  # number of EVs attending a ride req
    for v in vehicles:
        if isinstance(v.get_request(), RideRequest):
            riding_ev_count += 1
    ev_ride_time.append(riding_ev_count)
    ev_charge_time.append(Vehicle.ev_charging_count)
    ev_idle_time.append(n_veh - (riding_ev_count + Vehicle.ev_charging_count))

    # Generate a ride request
    for i in range(num_ride_req):
        # Create a ride request
        ride_req = RideRequest(PULoc[i], DOLoc[i], k, nx.shortest_path(g, source=PULoc[i], target=DOLoc[i]))
        req_vec.insert(req_idx, ride_req)

        # Compute cost of reaching the pickup point for each vehicle
        for j, veh in enumerate(vehicles):
            start_path[j] = nx.shortest_path(g, source=veh.get_position(),
                                             target=ride_req.get_origin())  # Path from vehicle node to request origin
            if not veh.is_available() or \
                    veh.get_soc() < min_consume + (
                    len(start_path[j]) + len(ride_req.get_path()) - 2) * travel_edge_time * discharge_rate or \
                    veh.get_soc() < Vehicle.min_charge or \
                    veh.is_charging() or \
                    delta_ride < len(start_path[j]) - 1:
                start_costs[j] = infeasib_cost
            else:
                start_costs[j] = len(start_path[j]) - 1

        cost.append(start_costs.copy())
        paths.append(start_path.copy())
        req_idx += 1

    if not req_idx:
        print("No real request at this time!")
        y_power_cars.append(Vehicle.ev_charging_count * power_transferred)
        miss_ride_time.append(0)
        continue
    elif sum([sum(cost[i]) for i in range(req_idx)]) == infeasib_cost * n_veh * req_idx:
        print("WARNING: No feasible request at this time!")
        y_power_cars.append(Vehicle.ev_charging_count * power_transferred)
        miss_ride_time.append(req_idx)
        continue

    # Add virtual requests or vehicles to make the assignment matrix square
    n_assign = max(req_idx, n_veh)
    if req_idx < n_veh:
        virtual_req_costs = [[infeasib_cost for col in range(n_veh)] for row in range(n_veh - req_idx)]
        cost = np.vstack([cost, virtual_req_costs]).transpose()
    elif req_idx > n_veh:
        virtual_req_costs = [[infeasib_cost for col in range(req_idx)] for row in range(req_idx - n_veh)]
        cost = np.vstack([np.array(cost).transpose(), virtual_req_costs])
    else:
        cost = np.array(cost).transpose()

    # Solve assignment problem, only riding part
    Kout = 2  # Iterations

    xi = [[0 for col in range(Kout + 1)] for row in range(n_assign * n_assign)]
    # Initialize with xi inside the feasible set, i.e. satisfies the constraints.
    for i in range(n_assign * n_assign):
        xi[i][0] = in_value

    cost_function = np.zeros(n_assign * n_assign)

    for t in range(Kout):
        for jj in range(n_assign):  # Loop over requests - columns
            for ii in range(n_assign):  # Loop over vehicles - rows
                req_j = ii * (1 + jj) + (n_assign - ii) * jj
                cost_function[req_j] = cost[ii][jj]

        # Solve using PuLP, a Python toolbox
        prob = pulp.LpProblem("AssignmentProblem", pulp.LpMinimize)

        x_list = []
        for i in range(n_assign * n_assign):
            x_list.append(pulp.LpVariable("x" + str(i), 0, 1, pulp.LpInteger))

        prob += pulp.lpSum(cost_function[m] * x_list[m] for m in range(n_assign * n_assign)), "obj function"

        for i in range(n_assign):
            prob += pulp.lpSum(x_list[i * n_assign + n] for n in range(n_assign)) == 1, "c_eq" + str(
                i)  # sum column elements
            prob += pulp.lpSum(x_list[i + n_assign * n] for n in range(n_assign)) <= 1, "c_ineq" + str(
                i)  # sum row elements

        prob.solve(pulp.COIN_CMD(msg=False))

        for v in prob.variables():
            index = int(v.name[1:])
            xi[index][t + 1] = v.varValue

    x_fin = np.array(xi)[:, -1]

    X = np.array(x_fin).reshape(n_assign, n_assign).transpose()
    cost_mat = cost_function.reshape(n_assign, n_assign).transpose()
    C_X = X * cost_mat

    # Assign requests
    for i in range(n_assign):
        veh_idx = X[:, i].tolist().index(1)
        if C_X[veh_idx][i] > infeasib_threas:  # Skip unfeasible requests
            continue
        vehicles[veh_idx].assign_request(req_vec[i])  # Assign request to vehicle
        if isinstance(req_vec[i], RideRequest):
            paths[i][veh_idx].pop()
            vehicles[veh_idx].assign_path(paths[i][veh_idx] + req_vec[i].get_path())  # Assign path (pickup + ride)
            vehicles[veh_idx].set_estimated_arrival(
                k + (len(vehicles[veh_idx].get_total_path()) - 1) * travel_edge_time + min_travel_time)
        else:
            raise ValueError("Wrong request assignment")

    y_power_cars.append(Vehicle.ev_charging_count * power_transferred)

    count_assigned_rides = 0
    for v in vehicles:
        if v.get_request() in req_vec and isinstance(v.get_request(), RideRequest):
            count_assigned_rides += 1
    if count_assigned_rides == req_idx:
        miss_ride_time.append(0)
    else:
        miss_ride_time.append(req_idx - count_assigned_rides)

    # Test
    for i in range(len(low_battery_time)):
        if not low_battery_time[i] + int_battery_time[i] + high_battery_time[i] == n_veh:
            raise ValueError("Wrong soc estimate")

# Results
time_slot = 15  # 15-min time slots
print("  --- Vehicles with low battery: ", low_battery_time[-1])
print("  --- Vehicles with int battery: ", int_battery_time[-1])
print("  --- Vehicles with high battery: ", high_battery_time[-1])
print("Missed ride-req, sum min by min: ", sum(miss_ride_time))
print("QoS: ", 100 - (sum(miss_ride_time) / tData.tot_num_requests_red * 100))

# Save results for later
path = ''
np.save(path + 'h_format', h_format)
np.save(path + 'miss_ride_time' + str(seed), miss_ride_time)
np.save(path + 'y_power_cars' + str(seed), y_power_cars)
np.save(path + 'high_battery_time' + str(seed), high_battery_time)
np.save(path + 'int_battery_time' + str(seed), int_battery_time)
np.save(path + 'low_battery_time' + str(seed), low_battery_time)
np.save(path + 'ev_ride_time' + str(seed), ev_ride_time)
np.save(path + 'ev_charge_time' + str(seed), ev_charge_time)
np.save(path + 'ev_idle_time' + str(seed), ev_idle_time)
