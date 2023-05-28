"""
Request class and child classes
"""


class Request:
    def __init__(self, origin, time_req):
        self.time_req = time_req
        self.origin = origin

    def get_origin(self):
        return self.origin


class RideRequest(Request):
    def __init__(self, origin, destination, time_req, path, bid=None, willing_to_share=None):
        super().__init__(origin, time_req)
        self.destination = destination
        self.path = path
        self.bid = bid
        self.willing_to_share = willing_to_share

    def get_destination(self):
        return self.destination

    def get_path(self):
        return self.path

    def is_willing_to_share(self):
        return self.willing_to_share

    def get_bid(self):
        return self.bid


class ChargeRequest(Request):
    def __init__(self, origin, time_req):
        super().__init__(origin, time_req)
