import heterocl as hcl
import numpy as np

class RelDyn_5D:
    """
    This class describe a 5D relative dynamics between a 4D bicycle model (robot car) and a 4D unicycle model (human car)
    The dynamics is defined as follows:

    x_rel' = (v_r/l_r) * sin(beta_r) * y_rel + v_h * cos(psi_rel) - v_r * cos(beta_r)
    y_rel' = (-v_r/l_r) * sin(beta_r) * x_rel + v_h * sin(psi_rel) - v_r * sin(beta_r)
    psi_rel' = w_h - (v_r/l_r) * sin(beta_r)
    v_h' = a_h
    v_r' = a_r
    beta_r = tan^-1(l_r/(l_f + l_r) * tan(delta_f))

    Controls: beta_r, a_r
    Disturbances: w_h, a_h

    In code:
    State : state = (state[0], state[1], state[2], state[3], state[4]) = (x_rel, y_rel, psi_rel, v_h, v_r)
    Control: uOpt = (uOpt[0], uOpt[1]) = (beta_r, a_r)
    Disturbance: dOpt = (dOpt[0], dOpt[1]) = (w_h, a_h)

    """
    def __init__(self, x=[0, 0, 0, 0, 0], uMin=np.array([-1, -1]), uMax=np.array([1, 1]), dMin=np.array([-1, -1]),
                 dMax=np.array([1, 1]), dims=5, uMode="max", dMode="min"):
        self.x = x
        self.uMode = uMode
        self.dMode = dMode

        # Object properties
        self.x = x

        # Control bounds
        self.uMin = uMin
        self.uMax = uMax

        # Disturbance bounds
        self.dMin = dMin
        self.dMax = dMax

        self.dims = dims

        # Some constants
        self.l_r = 2 * 1.738
        self.l_f = 2 * 1.058

    def dynamics(self, state, uOpt, dOpt):
        """

        :param t:
        :param state:
        :param uOpt:
        :param dOpt:
        :return:
        """
        x1_dot = hcl.scalar(0, "x1_dot")
        x2_dot = hcl.scalar(0, "x2_dot")
        x3_dot = hcl.scalar(0, "x3_dot")
        x4_dot = hcl.scalar(0, "x4_dot")
        x5_dot = hcl.scalar(0, "x5_dot")

        x1_dot[0] = (state[4] / self.l_r) * np.sin(uOpt[0]) * state[1] + state[3] * np.cos(state[2]) - state[4] * np.cos(uOpt[0])
        x2_dot[0] = (-state[4] / self.l_r) * np.sin(uOpt[0]) * state[0] + state[3] * np.sin(state[2]) - state[4] * np.sin(uOpt[0])
        x3_dot[0] = dOpt[0] - (state[4] / self.l_r) * np.sin(uOpt[0])
        x4_dot[0] = dOpt[1]
        x5_dot[0] = uOpt[1]

        return (x1_dot[0], x2_dot[0], x3_dot[0], x4_dot[0], x5_dot[0])

    def opt_ctrl(self, state, spat_deriv):
        """
        For all the notation here, please refer to doc "reachability for relative dynamics"

        :param state:
        :param spat_deriv:
        :return:
        """

        # uOpt1: beta_r, uOpt2: a_r
        uOpt1 = hcl.scalar(0, "uOpt1")
        uOpt2 = hcl.scalar(0, "uOpt2")

        # Define some constant
        c1 = hcl.scalar(0, "c1")
        c2 = hcl.scalar(0, "c2")

        # According to doc, c1, c2 are defined as follow
        c1[0] = spat_deriv[0] * (state[4] / self.l_r) * state[1] - spat_deriv[1] * (state[4] / self.l_r) * state[0] -\
                spat_deriv[1] * state[4] - spat_deriv[2] * (state[4] / self.l_r)
        c2[0] = - spat_deriv[0] * state[4]

        with hcl.if_(self.uMode == "max"):

            # For uOpt1: beta_r
            with hcl.if_(c1[0] > 0):
                with hcl.if_(self.dMin[0] <= (- np.arctan(c2 / c1) + np.pi / 2) <= self.dMax[0]):
                    uOpt1[0] = - np.arctan(c2 / c1) + np.pi / 2
                with hcl.if_((- np.arctan(c2 / c1) + np.pi / 2) > self.dMax[0]):
                    uOpt1[0] = self.dMax[0]
                with hcl.if_((- np.arctan(c2 / c1) + np.pi / 2) < self.dMin[0]):
                    uOpt1[0] = self.dMin[0]
            with hcl.if_(c1[0] < 0):
                with hcl.if_(self.dMin[0] <= (- np.arctan(c2 / c1) - np.pi / 2) <= self.dMax[0]):
                    uOpt1[0] = - np.arctan(c2 / c1) - np.pi / 2
                with hcl.if_((- np.arctan(c2 / c1) - np.pi / 2) > self.dMax[0]):
                    uOpt1[0] = self.dMax[0]
                with hcl.if_((- np.arctan(c2 / c1) - np.pi / 2) < self.dMin[0]):
                    uOpt1[0] = self.dMin[0]
            with hcl.if_(c1[0] == 0):
                with hcl.if_(c2[0] >= 0):
                    with hcl.if_(self.dMin[0] <= 0 <= self.dMax[0]):
                        uOpt1[0] = 0
                    with hcl.if_((0 < self.dMin[0]) or (0 > self.dMax[0])):
                        uOpt1[0] = np.min(np.abs(self.dMin[0]), np.abs(self.dMax[0]))
                with hcl.if_(c2[0] < 0):
                    with hcl.if_(np.abs(self.dMin[0]) >= np.abs(self.dMax[0])):
                        uOpt1[0] = np.abs(self.dMin[0])
                    with hcl.if_(np.abs(self.dMin[0]) < np.abs(self.dMax[0])):
                        uOpt1[0] = np.abs(self.dMax[0])

            # For uOpt2: a_r
            with hcl.if_(spat_deriv[4] > 0):
                uOpt2[0] = self.uMax[1]
            with hcl.if_(spat_deriv[4] <= 0):
                uOpt2[0] = self.uMin[1]

        return (uOpt1[0], uOpt2[0])

    def optDstb(self, state, spat_deriv):
        """

        :param state:
        :param spat_deriv:
        :return:
        """

        dOpt1 = hcl.scalar(0, "dOpt1")
        dOpt2 = hcl.scalar(0, "dOpt2")

        with hcl.if_(self.uMode == "max"):

            # For dOpt1: w_h
            with hcl.if_(spat_deriv[2] > 0):
                dOpt1[0] = self.dMin[0]
            with hcl.if_(spat_deriv[2] <= 0):
                dOpt1[0] = self.dMax[0]

            # For dOpt2: a_h
            with hcl.if_(spat_deriv[3] > 0):
                dOpt2[0] = self.dMin[1]
            with hcl.if_(spat_deriv[3] <= 0):
                dOpt2[0] = self.dMax[1]

        return (dOpt1[0], dOpt2[0])