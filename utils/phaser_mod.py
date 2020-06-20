import numpy as np
import scipy.signal as signal

class Phasor:
    def __init__(self, fs, Nstages=10):
        self.fs = fs
        self.z = np.zeros((Nstages,2))

    def calc_coefs(self, R):
        C = 25.0e-9
        RC = R*C
        b_s = np.array([RC, -1.0])
        a_s = np.array([RC, 1.0])

        b, a = signal.bilinear(b_s, a_s, fs=self.fs)
        return b, a

    def process_stage(self, x, b, a, stage):
        y = self.z[stage][1] + x * b[0]
        self.z[stage][1] = x*b[1] - y*a[1]
        return y

    def process_block(self, x, lfo, Nstages):
        y = np.copy(x)
        max_depth = 20.0
        for n in range(len(x)):
            light_val = (max_depth + 0.1) - (lfo[n] * max_depth)
            r_val = 100000 * (light_val / 0.1)**(-0.75)
            b, a = self.calc_coefs(r_val)

            for stage in range(Nstages):
                x[n] = self.process_stage(x[n], b, a, stage)

            y[n] = x[n]

        return y
