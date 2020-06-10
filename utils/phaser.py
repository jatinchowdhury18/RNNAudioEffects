import numpy as np
import scipy.signal as signal

class Phasor:
    def __init__(self, fs):
        self.fs = fs
        self.z = np.zeros(3)

    def calc_coefs(self, R, fbAmt):
        C = 15.0e-9
        RC = R*C
        b_s = np.array([RC*RC, -2*RC, 1.0])
        a_s = np.array([b_s[0] * (1.0 + fbAmt), -b_s[1] * (1.0 - fbAmt), 1.0 + fbAmt])

        b, a = signal.bilinear(b_s, a_s, fs=self.fs)
        return b, a

    def process_sample(self, x, b, a):
        y = self.z[1] + x * b[0]
        self.z[1] = self.z[2] + x*b[1] - y*a[1]
        self.z[2] = x*b[2] - y*a[2]
        return y

    def process_block(self, x, lfo, fb_amt):
        y = np.copy(x)
        max_depth = 20.0
        for n in range(len(x)):
            light_val = (max_depth + 0.1) - (lfo[n] * max_depth)
            r_val = 100000 * (light_val / 0.1)**(-0.75)
            b, a = self.calc_coefs(r_val, fb_amt)
            y[n] = self.process_sample(x[n], b, a)

        return y
