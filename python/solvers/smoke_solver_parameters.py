
class SmokeSolverParameters:
    def __init__(self):
        self.alp = 10.0
        self.bet = 10.0
        self.temp_decay = 0.05

    def set_alpha(self, alpha):
        self.alp = alpha

    def set_beta(self, beta):
        self.bet = beta

    def set_temperature_decay(self, t_decay):
        self.temp_decay = t_decay

    def temperature_decay(self):
        return self.temp_decay

    def alpha(self):
        return self.alp

    def beta(self):
        return self.bet

    def no_temperature(self):
        self.alp = 0.0
        self.bet = 0.0
