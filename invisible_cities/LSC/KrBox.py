class KrBox:
    """Container of krypton events"""
    def __init__(self, run_number):
        self.run_number = run_number
        self.event_ = []
        self.s1f_ = S12F()
        self.s2f_ = S12F()
        self.qs1_ = []
        self.qs2_ = []
        self.drift_time_ = []
        self.X_ = []
        self.Y_ = []
        self.Z_ = []
        self.R_ = []
        self.Phi_ = []

    def add_position(self, X, Y, Z, R, Phi, drift_time):
        self.X_.append(X)
        self.Y_.append(Y)
        self.Z_.append(Z)
        self.R_.append(R)
        self.Phi_.append(Phi)
        self.drift_time_.append(drift_time)

    def event(self):
        return np.array(self.event_)
    def s1f(self):
        return np.array(self.s1f)
    def s2f(self):
        return np.array(self.s2f)
    def qs1(self):
        return np.array(self.qs1)
    def qs2(self):
        return np.array(self.qs2)
    def drift_time(self):
        return np.array(self.drift_time_)
    def X(self):
        return np.array(self.X_)
    def Y(self):
        return np.array(self.Y_)
    def Z(self):
        return np.array(self.Z_)
    def R(self):
        return np.array(self.R_)
    def Phi(self):
        return np.array(self.Phi_)
