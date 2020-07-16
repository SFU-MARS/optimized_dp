import heterocl as hcl
import numpy as np
import time

import computeGraphs
"""
4D BICYCLE DYNAMICS IMPLEMENTATION
x_dot = v * cos(psi + beta)
y_dot = v * sin(psi + beta)
v_dot = a
psi_dot = (v / l_r) * sin(beta)
beta = tan^(-1) * (l_r / (l_f + l_r) * tan(delta_f))

control: u = (a, delta_f)


"""
class Bicycle4D:
    def __init__(self, x=[0, 0, 0, 0], uMax=[], uMin=[], uMode="min", dMode="max"):
        self.x = x
        self.uMax = uMax
        self.uMin = uMin
        self.uMode = uMode
        self.dMode = dMode

    def opt_ctrl(self, t, state, spat_deriv):
        """
        :param t: time t
        :param state: tuple of coordinates
        :param spat_deriv: tuple of spatial derivative in all dimensions
        :return:
        """

