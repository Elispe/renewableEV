"""
Vehicle class
"""


class Vehicle:
    min_charge = 6
    int_charge = 31
    full_charge = 50  # kWh
    ev_charging_count = 0  # number of EVs currently charging
    ev_capacity = 4  # max number of passengers

    def __init__(self, position, soc):
        self.position = position
        self.soc = soc
        self.capacity = Vehicle.ev_capacity
        self.path = []
        self.last_update = 0
        self.charging = False
        self.request = None
        self.estimated_charged = -1
        self.estimated_arrival = -1
        self.removed = True

    def __str__(self):
        return f"Vehicle at node {self.position} with soc {self.soc}"

    def set_soc(self, new_soc):
        self.soc = new_soc

    def get_soc(self):
        return self.soc

    def set_position(self, new_position):
        self.position = new_position

    def get_position(self):
        return self.position

    def reset_capacity(self):
        self.capacity = Vehicle.ev_capacity

    def decrease_capacity(self):
        self.capacity -= 1

    def get_capacity(self):
        return self.capacity

    def assign_request(self, request):
        self.request = request

    def assign_path(self, path):
        self.path = path

    def get_total_path(self):
        return self.path

    def is_available(self):
        if self.request is None:
            return True
        else:
            return False

    def is_charging(self):
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

    def terminate_request(self):
        self.request = None

    def get_request(self):
        return self.request

    def get_estimated_arrival(self):
        return self.estimated_arrival

    def set_estimated_arrival(self, estimated_arrival):
        self.estimated_arrival = estimated_arrival

    def get_estimated_charged(self):
        return self.estimated_charged

    def set_estimated_charged(self, estimated_charged):
        self.estimated_charged = estimated_charged

    def is_removed(self):
        return self.removed

    def remove(self):
        if self.removed:
            self.removed = False
        else:
            self.removed = True

    def shorten_path(self):
        self.path.pop(0)

    def update(self, time):
        self.last_update = time

    def get_last_update(self):
        return self.last_update
