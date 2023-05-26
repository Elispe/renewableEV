"""
Request class and child classes
"""


class Request:
    def __init__(self, destination, time_req):
        self.time_req = time_req
        self.destination = destination

    def getDestination(self):
        return self.destination

class RideRequest(Request):
    def __init__(self, start, destination, time_req, path):
        super().__init__(destination, time_req)
        self.start = start
        self.path = path

    def getPickupPoint(self):
        return self.start

    def getPath(self):
        return self.path

class ChargeRequest(Request):
    def __init__(self, destination, time_req):
        super().__init__(destination, time_req)