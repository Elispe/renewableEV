"""
Vehicle class
"""


class Vehicle:
    min_charge = 6
    int_charge = 31
    full_charge = 50 # kWh
    ev_charging_count = 0

    def __init__(self, position, soc):
        self.position = position
        self.soc = soc

    def __str__(self):
        return f"Vehicle at node {self.position} with soc {self.soc}"

    def setSoc(self, newSoc):
        self.soc = newSoc

    def getSoc(self):
        return self.soc

    def setPosition(self, newPosition):
        self.position = newPosition

    def getPosition(self):
        return self.position

    def setCapacity(self, capacity):
        self.capacity = capacity

    def decreaseCapacity(self):
        self.capacity -= 1

    def assignRequest(self, request):
        self.request = request

    def assignPath(self, path):
        self.path = path

    def getTotalPath(self):
        return self.path

    def isAvailable(self):
        if not hasattr(self, "request") or self.request is None:
            return True
        else:
            return False

    def isCharging(self):
        if not hasattr(self, "charging"):
            return False
        else:
            return self.charging

    def charge(self, charge_rate=None):
        if charge_rate is None:
            self.charging = True
            Vehicle.ev_charging_count += 1
        else:
            self.soc += charge_rate

        if self.soc >= Vehicle.full_charge:
            self.soc = Vehicle.full_charge
            self.charging = False
            Vehicle.ev_charging_count -= 1

    def discharge(self, discharge_rate):
        self.soc -= discharge_rate
        if self.soc < 0:
            raise ValueError("Negative SOC")

    def terminateRequest(self):
        self.request = None

    def getRequest(self):
        return self.request
    def getEstimatedArrival(self):
        if not hasattr(self, "estimated_arrival"):
            return -1
        else:
            return self.estimated_arrival

    def setEstimatedArrival(self, estimated_arrival):
        self.estimated_arrival = estimated_arrival




