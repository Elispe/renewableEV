# This is the main Python script for case1 and case2.
# Both ride and charge requests are sent to the ride-service provider.
# Include ride-sharing option

import numpy as np
import networkx as nx
import pulp
import tripData as tData
import time
import cvxpy as cp
import datetime
import sys

from Vehicle import Vehicle
from Request import RideRequest
from Request import ChargeRequest

# Customer's willingness to share a ride:
sharing_prob = 0.75  # Enter value between 0 and 1

# Seed corresponds to SLURM_ARRAY_TASK_ID
seed = int(sys.argv[1])
np.random.seed(seed)

# Fleet size
n_veh = 100

# generation profile, select data to import depending on time window
sunny_1day = np.array(tData.pv_sunny)
# cloud_am_1day = np.array(tData.pv_cloud_am)
# cloud_pm_1day = np.array(tData.pv_cloud_pm)
# charge_req_prob = sunny_1day
# if simulation window runs after midnight
charge_req_prob = np.concatenate((sunny_1day, sunny_1day))

# Variable initialization
delta_ride = 2  # Max zones away from customer for pick-up
delta_charge = 1  # Max zones away from charging station
power_transferred = 12  # kW for each car charging
discharge_rate = 0.1  # kWh/minute
charge_rate = discharge_rate * 2
min_travel_time = 5  # min
min_consume = min_travel_time * discharge_rate
infeasib_cost = 1e5
infeasib_threas = 1e4
travel_edge_time = 10
in_value = 1.0 / n_veh  # for initialization of x_ij
station_power = 25  # kW generated at peak power

# Random SOC array
random_soc = [np.random.uniform(Vehicle.min_charge, Vehicle.full_charge) for n in range(n_veh)]
random_noise = [np.random.binomial(1, charge_req_prob[minu]) for minu in range(24 * tData.tot_day * 60)]

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

# Randomize the initial soc, otherwise comment below
for j, veh in enumerate(vehicles):
    veh.set_soc(random_soc[j])

# Track charging
y_Pref_current = []
y_power_cars = []
station_nodes = [3, 5, 8, 9]  # nodes where charging facilities exist
cars_charging = [0] * len(station_nodes)
future_cars_charging = [0] * len(station_nodes)

# Variables to track performance
miss_ride_time = []
low_battery_time = []
int_battery_time = []
high_battery_time = []
ev_ride_time = []
ev_charge_time = []
ev_idle_time = []
h_format = []
incent_ride_assigned = []
incent_charge_assigned = []

for k in range(tData.num_min):
    minute = k + tData.h_in * 60
    PULoc = tData.records[k][0]
    DOLoc = tData.records[k][1]
    num_ride_req = len(PULoc)
    h_bid = [np.random.uniform(tData.h_aux_min[minute], tData.h_aux[minute]) for i in range(num_ride_req)]
    b_min = [np.random.uniform(-tData.b_aux[minute], 0) for i in range(len(station_nodes))]
    b_max = [np.random.uniform(0, tData.b_aux[minute]) for i in range(len(station_nodes))]
    c_RES = tData.c_RES_total[minute]
    alpha_w = tData.alpha_w_total[minute]
    beta_w = tData.beta_w_total[minute]
    h_format.append(time.strftime("%H:%M", time.gmtime(minute * 60)))
    # to run over multiple days use code below instead
    # dt = datetime.datetime(2022, 3, 1) - datetime.datetime(1970, 1, 1)
    # minutessince = int(dt.total_seconds() / 60)
    # h_format.append(time.strftime("%b %d %H:%M", time.gmtime((minutessince + minute) * 60)))

    print("*** Minute: " + str(k) + " ***")

    req_vec = []  # List of requests collected between time t and t+1
    req_idx = 0
    ride_req_idx = 0
    charge_req_idx = 0

    start_costs = [0] * n_veh  # costs to get to the pick-up point
    start_path = [0] * n_veh  # path to get to the pick-up point
    start_costs_charge = [0] * n_veh  # costs to get to charging station
    start_path_charge = [0] * n_veh  # path to get to the charging station

    cost = []
    paths = []

    # Update vehicle positions
    for veh in vehicles:
        if len(veh.get_total_path()) >= 2 and (k - veh.get_last_update()) // travel_edge_time == 1:
            veh.update(k)
            veh.shorten_path()
            veh.set_position(veh.get_total_path()[0])  # Update the vehicle position to next node

    for veh in vehicles:
        # If vehicle reaches passenger final destination
        if not veh.is_available() and isinstance(veh.request,
                                                 RideRequest) and veh.get_estimated_arrival() <= k and \
                veh.get_position() == veh.get_request().get_destination() and len(veh.get_total_path()) == 1:  # check
            veh.terminate_request()  # Vehicle again available
            veh.reset_capacity()
        # If vehicle reaches charging station
        if not veh.is_available() and isinstance(veh.request, ChargeRequest) and \
                veh.get_position() == veh.get_request().get_origin() and len(veh.get_total_path()) == 1:
            if veh.get_estimated_arrival() < k and not veh.is_charging():  # to check
                veh.charge()
                cars_charging[station_nodes.index(veh.get_position())] += 1
            elif veh.get_estimated_charged() - min_travel_time <= k and veh.is_charging() and not veh.is_removed():
                future_cars_charging[station_nodes.index(veh.get_request().get_origin())] -= 1
                veh.remove()

    # Update vehicle state-of-charge
    # If vehicle is fully charged, disconnect
    for veh in vehicles:
        if veh.get_estimated_arrival() >= k and not veh.is_charging():
            veh.discharge(discharge_rate)
        elif veh.is_charging():
            veh.charge(charge_rate)
            if veh.soc >= Vehicle.full_charge:
                cars_charging[station_nodes.index(veh.get_request().get_origin())] -= 1
                veh.terminate_request()  # Vehicle again available

    # Track SOC status
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
    charging_ev_count = 0  # number of EVs attending a charge req
    for v in vehicles:
        if isinstance(v.get_request(), RideRequest):
            riding_ev_count += 1
        elif isinstance(v.get_request(), ChargeRequest):
            charging_ev_count += 1
    ev_ride_time.append(riding_ev_count)
    ev_charge_time.append(charging_ev_count)
    ev_idle_time.append(n_veh - (riding_ev_count + charging_ev_count))

    # Generate a ride request
    for i in range(num_ride_req):
        # Create a ride request
        ride_req = RideRequest(PULoc[i], DOLoc[i], k, nx.shortest_path(g, source=PULoc[i], target=DOLoc[i]), h_bid[i],
                               np.random.binomial(1, sharing_prob))
        req_vec.insert(req_idx, ride_req)

        # Compute cost of reaching the pickup point for each vehicle
        for j, veh in enumerate(vehicles):
            start_path[j] = nx.shortest_path(g, source=veh.get_position(),
                                             target=ride_req.get_origin())  # Path from vehicle node to request origin
            if isinstance(veh.get_request(), ChargeRequest) or \
                    veh.get_soc() < min_consume + (
                    len(start_path[j]) + len(ride_req.get_path()) - 2) * travel_edge_time * discharge_rate or \
                    veh.get_soc() < Vehicle.min_charge or \
                    delta_ride < len(start_path[j]) - 1:
                start_costs[j] = infeasib_cost
            elif isinstance(veh.get_request(), RideRequest):  # check if ride can be shared
                if start_path[j][:-1] + ride_req.get_path() == veh.get_total_path() and veh.get_capacity() > 0 and \
                        k - veh.get_last_update() < travel_edge_time // 2 and veh.get_request().is_willing_to_share() \
                        and ride_req.is_willing_to_share():
                    start_costs[j] = len(start_path[j]) - 1
                else:
                    start_costs[j] = infeasib_cost
            else:
                start_costs[j] = len(start_path[j]) - 1

        cost.append(start_costs.copy())
        paths.append(start_path.copy())
        ride_req_idx += 1
        req_idx += 1

    # Generate one or more charge requests
    PrefMax = np.array([12, 19, 6, 2]) * station_power
    y_Pref_current.append(PrefMax * charge_req_prob[minute])

    PrefAvailable = []
    for i in range(len(station_nodes)):
        PrefAvailable.append(
            round(charge_req_prob[minute] * PrefMax[i] - power_transferred * future_cars_charging[i]))

    Pref = 0
    num_req_station = []
    for p in PrefAvailable:
        if p >= Vehicle.full_charge // 2 and random_noise[minute]:
            Pref += p
            charge_loc = station_nodes[PrefAvailable.index(p)]
            numChargeReq = p // (Vehicle.full_charge // 2)
            num_req_station.append(numChargeReq)
            for i in range(numChargeReq):
                # Create a charge request
                charge_req = ChargeRequest(charge_loc, k)
                req_vec.insert(req_idx, charge_req)
                charge_req_idx += 1
                req_idx += 1

                # Compute cost of reaching the charging station point for each vehicle
                for j, veh in enumerate(vehicles):
                    start_path_charge[j] = nx.shortest_path(g, source=veh.get_position(),
                                                            target=charge_req.get_origin())  # Path to charging station
                    if not veh.is_available() or veh.get_soc() > 2 / 3 * Vehicle.full_charge or \
                            veh.get_soc() < min_consume + (
                            len(start_path_charge[j]) - 1) * travel_edge_time * discharge_rate or \
                            delta_charge < len(start_path_charge[j]) - 1:
                        start_costs_charge[j] = infeasib_cost
                    else:
                        start_costs_charge[j] = len(start_path_charge[j]) - 1

                cost.append(start_costs_charge.copy())
                paths.append(start_path_charge.copy())
        else:
            num_req_station.append(0)

    if not req_idx:
        print("No real request at this time!")
        y_power_cars.append(np.array(cars_charging) * power_transferred)
        miss_ride_time.append(0)
        incent_ride_assigned.append(0)
        incent_charge_assigned.append(0)
        continue
    elif sum([sum(cost[i]) for i in range(req_idx)]) == infeasib_cost * n_veh * req_idx:
        print("WARNING: No feasible request at this time!")
        y_power_cars.append(np.array(cars_charging) * power_transferred)
        incent_ride_assigned.append(0)
        incent_charge_assigned.append(0)
        if ride_req_idx:
            miss_ride_time.append(ride_req_idx)
        else:
            miss_ride_time.append(0)
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

    cost_array = cost.transpose().flatten()
    inf_idx = [i for i, x in enumerate(cost_array) if x == infeasib_cost]
    feas_idx_num = n_assign * n_assign - len(inf_idx)

    # Solve assignment problem
    Kout = 2  # Iterations outer loop
    Kin = 50  # Iterations inner loop

    yStar = [0] * n_assign * n_assign

    skip = 0
    for kreq in range(ride_req_idx):
        for kk in range(n_veh):
            yStar[kk + (n_veh + skip) * kreq] = req_vec[kreq].get_bid() - alpha_w * (
                    len(req_vec[kreq].get_path()) + len(paths[kreq][kk]) - 2) + beta_w * (
                                                        Vehicle.ev_capacity - vehicles[kk].get_capacity())
        if n_veh < req_idx:
            skip = req_idx - n_veh

    for kreq in range(ride_req_idx, req_idx):  # to do: remove these incentives
        for kk in range(n_veh):
            yStar[kk + (n_veh + skip) * kreq] = 0.01 * (Vehicle.full_charge - vehicles[kk].get_soc())
        if n_veh < req_idx:
            skip = req_idx - n_veh

    y = [[0 for col in range(Kout)] for row in range(n_assign * n_assign)]
    xi = [[0 for col in range(Kout + 1)] for row in range(n_assign * n_assign)]
    # Initialize with xi inside the feasible set, i.e. satisfies the constraints.
    for i in range(n_assign * n_assign):
        xi[i][0] = in_value

    cost_function = np.zeros(n_assign * n_assign)
    yStar_feas_ride = []
    for jj in range(ride_req_idx):  # Loop over requests - columns
        for ii in range(n_veh):  # Loop over vehicles - rows
            req_j = ii * (1 + jj) + (n_assign - ii) * jj
            if req_j not in inf_idx:
                yStar_feas_ride.append(yStar[req_j])

    yStar_feas_charge = []
    for s in range(len(station_nodes)):
        yStar_feas_ch_s = []
        ind_in = sum(num_req_station[0:s], ride_req_idx)
        ind_fin = sum(num_req_station[0:s + 1], ride_req_idx)
        for jj in range(ind_in, ind_fin):  # Loop over charge requests at station s - columns
            for ii in range(n_veh):  # Loop over vehicles - rows
                req_j = ii * (1 + jj) + (n_assign - ii) * jj
                if req_j not in inf_idx:
                    yStar_feas_ch_s.append(yStar[req_j])
        yStar_feas_charge.append(yStar_feas_ch_s)

    for t in range(Kout):
        x_feas_ride = []
        for jj in range(ride_req_idx):  # Loop over ride requests - columns
            for ii in range(n_veh):  # Loop over vehicles - rows
                req_j = ii * (1 + jj) + (n_assign - ii) * jj
                if req_j not in inf_idx:
                    x_feas_ride.append(xi[req_j][t])

        x_feas_ch = []
        for s in range(len(station_nodes)):
            x_feas_ch_s = []
            ind_in = sum(num_req_station[0:s], ride_req_idx)
            ind_fin = sum(num_req_station[0:s + 1], ride_req_idx)
            for jj in range(ind_in, ind_fin):  # Loop over charge requests at station s - columns
                for ii in range(n_veh):  # Loop over vehicles - rows
                    req_j = ii * (1 + jj) + (n_assign - ii) * jj
                    if req_j not in inf_idx:
                        x_feas_ch_s.append(xi[req_j][t])
            x_feas_ch.append(x_feas_ch_s)

        # Solve inner using cvxpy
        fin_incent_y = []
        # ride req part
        if len(x_feas_ride):
            yinn_ride = cp.Variable(len(x_feas_ride))
            objective = cp.Minimize(cp.sum_squares(yinn_ride - np.array(yStar_feas_ride)))
            constraint = [-1 <= yinn_ride, yinn_ride <= 1]
            problem = cp.Problem(objective, constraint)
            problem.solve()
            fin_incent_y = yinn_ride.value.tolist()

            if t == Kout - 1:
                if np.sum(x_feas_ride) != 0:
                    incent_ride_assigned.append(
                        np.dot(np.array(x_feas_ride), np.array(fin_incent_y)) / np.sum(x_feas_ride))
                else:
                    incent_ride_assigned.append(0)
        else:
            if t == Kout - 1:
                incent_ride_assigned.append(0)

        # charge req part
        sum_charge_inc = 0
        for s in range(len(station_nodes)):
            if len(x_feas_ch[s]):
                yinn_ch = cp.Variable(len(x_feas_ch[s]))
                obj = cp.Minimize((PrefAvailable[s] * c_RES - yinn_ch @ np.array(x_feas_ch[s]) ** 2))
                constr = [yinn_ch @ np.array(x_feas_ch[s]) >= b_min[s],
                          yinn_ch @ np.array(x_feas_ch[s]) <= b_max[s], -1 <= yinn_ch, yinn_ch <= 1]
                probl = cp.Problem(obj, constr)
                probl.solve()
                fin_incent_y = np.concatenate((fin_incent_y, yinn_ch.value)).tolist()

                if t == Kout - 1:
                    if np.sum(x_feas_ch[s]) != 0:
                        sum_charge_inc += np.dot(np.array(x_feas_ch[s]), np.array(yinn_ch.value)) / np.sum(
                            x_feas_ch[s])

        if t == Kout - 1:
            incent_charge_assigned.append(sum_charge_inc)

        # Cost function for the assignment problem - based on original cost and price incentives
        for jj in range(n_assign):  # Loop over requests - columns
            for ii in range(n_assign):  # Loop over vehicles - rows
                req_j = ii * (1 + jj) + (n_assign - ii) * jj
                if req_j not in inf_idx:
                    cost_function[req_j] = cost[ii][jj] - fin_incent_y.pop(0)
                else:
                    cost_function[req_j] = cost[ii][jj]

        if fin_incent_y:
            raise Exception("ERROR: incorrect number of incentives")

        # Solve the assignment problem using PuLP, a Python toolbox
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
        vehicles[veh_idx].update(k)
        if isinstance(req_vec[i], RideRequest):
            vehicles[veh_idx].decrease_capacity()
            paths[i][veh_idx].pop()
            vehicles[veh_idx].assign_path(paths[i][veh_idx] + req_vec[i].get_path())  # Assign path (pickup + ride)
            vehicles[veh_idx].set_estimated_arrival(
                k + (len(vehicles[veh_idx].get_total_path()) - 1) * travel_edge_time + min_travel_time)
        elif isinstance(req_vec[i], ChargeRequest):
            vehicles[veh_idx].assign_path(paths[i][veh_idx])  # Assign path to reach the charging station
            # car discharges even more to reach the station
            ai = Vehicle.full_charge - vehicles[veh_idx].get_soc() + (len(
                vehicles[veh_idx].get_total_path()) - 1) * travel_edge_time * discharge_rate + min_consume
            vehicles[veh_idx].set_estimated_arrival(
                k + (len(vehicles[veh_idx].get_total_path()) - 1) * travel_edge_time + min_travel_time)
            vehicles[veh_idx].set_estimated_charged(vehicles[veh_idx].get_estimated_arrival() + int(
                round(ai / power_transferred * 60)))
            future_cars_charging[station_nodes.index(vehicles[veh_idx].get_request().get_origin())] += 1
            vehicles[veh_idx].remove()
        else:
            print("Wrong request assignment")

    y_power_cars.append(np.array(cars_charging) * power_transferred)

    count_assigned_rides = 0
    for v in vehicles:
        if v.get_request() in req_vec and isinstance(v.request, RideRequest):
            count_assigned_rides += 1

    if count_assigned_rides == ride_req_idx:
        miss_ride_time.append(0)
    else:
        miss_ride_time.append(ride_req_idx - count_assigned_rides)

    # Test
    for i in range(len(low_battery_time)):
        if not low_battery_time[i] + int_battery_time[i] + high_battery_time[i] == n_veh:
            raise Exception("ERROR: wrong soc estimate")

# Results
time_slot = 15  # 15-min time slots
print("  --- Vehicles with low battery: ", low_battery_time[-1])
print("  --- Vehicles with int battery: ", int_battery_time[-1])
print("  --- Vehicles with high battery: ", high_battery_time[-1])
print("Missed ride-req, sum min by min: ", sum(miss_ride_time))
print("QoS: ", 100 - (sum(miss_ride_time) / tData.tot_num_requests_red * 100))
lost_power_percent = 0
for i in range(len(y_Pref_current)):
    if sum(y_Pref_current[i]) > sum(y_power_cars[i]):
        lost_power_percent += (sum(y_Pref_current[i]) - sum(y_power_cars[i]))

tot_Pref = sum([sum(i) for i in y_Pref_current])
print("Power lost: " + str(round((lost_power_percent / tot_Pref) * 100, 2)) + "%")

# Save results for later
path = ''
np.save(path + 'h_format', h_format)
np.save(path + 'numReq', tData.tot_num_requests_red)
np.save(path + 'y_Pref_current', y_Pref_current)
np.save(path + 'miss_ride_time' + str(seed), miss_ride_time)
np.save(path + 'y_power_cars' + str(seed), y_power_cars)
np.save(path + 'high_battery_time' + str(seed), high_battery_time)
np.save(path + 'int_battery_time' + str(seed), int_battery_time)
np.save(path + 'low_battery_time' + str(seed), low_battery_time)
np.save(path + 'ev_ride_time' + str(seed), ev_ride_time)
np.save(path + 'ev_charge_time' + str(seed), ev_charge_time)
np.save(path + 'ev_idle_time' + str(seed), ev_idle_time)
np.save(path + 'incent_charge_assigned' + str(seed), incent_charge_assigned)
np.save(path + 'incent_ride_assigned' + str(seed), incent_ride_assigned)
