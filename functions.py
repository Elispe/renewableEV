import networkx as nx

# Import classes
from Request import RideRequest
from Request import ChargeRequest

# Functions
def create_ride_request(start, end, time, graph):
    start = start
    end = end
    time_req = time  # Time at which the request is placed
    path = nx.shortest_path(graph, source=start, target=end)
    return RideRequest(start, end, time_req, path)

def create_charge_request(location, time):
    location = location
    time_req = time
    return ChargeRequest(location, time_req)
