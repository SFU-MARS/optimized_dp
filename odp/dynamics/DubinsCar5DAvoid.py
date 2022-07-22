import heterocl as hcl
import numpy as np

"""
DubinsCar5D avoidance System 
 x_1_dot = -x_4 + x_5*cos(x_3) + u_theta * x_2 
 x_2_dot = x_5*sin(x_3) - u_theta * x_1
 x_3_dot = d_theta - u_theta
 x_4_dot = u_v
 x_5_dot = d_v
 |u_theta| <= u_theta_max
 |u_v| <= u_v_max
 |d_theta| <= d_theta_max
 |d_v| <= d_v_max 
"""
def my_sign(x):
    sign = hcl.scalar(0, "sign", dtype=hcl.Float())
    with hcl.if_(x == 0):
        sign[0] = 0
    with hcl.if_(x > 0):
        sign[0] = 1
    with hcl.if_(x < 0):
        sign[0] = -1
    return sign[0]

class DubinsCar5DAvoid:
    def __init__(self, x=[0,0,0,0,0], u_theta_max = 1, u_v_max = 1, d_theta_max=1, d_v_max=1, uMode="min", dMode="max"):
        self.x = x
        self.u_theta_max = u_theta_max
        self.u_v_max = u_v_max
        self.d_theta_max = d_theta_max
        self.d_v_max = d_v_max
        self.uMode = uMode
        self.dMode = dMode

    def opt_ctrl(self, t, state, spat_deriv):
        opt_u_theta = hcl.scalar(0, "opt_u_theta")
        opt_u_v = hcl.scalar(0, "opt_u_v")
        # Just create and pass back, even though they're not used
        in3   = hcl.scalar(0, "in3")
        in4   = hcl.scalar(0, "in4")
        in5   = hcl.scalar(0, "in5")

        with hcl.if_(self.uMode == "max"):
            # use computeGraphs/CustomGraphFunctions#my_sign instead of sign
            opt_u_theta[0] = my_sign(spat_deriv[0]*self.x[1] - spat_deriv[1]*self.x[0] - spat_deriv[2]) * self.u_theta_max
            opt_u_v[0] = my_sign(spat_deriv[3]) * self.u_v_max

        with hcl.if_(self.uMode == "min"):
            opt_u_theta[0] = -my_sign(spat_deriv[0]*self.x[1] - spat_deriv[1]*self.x[0] - spat_deriv[2]) * self.u_theta_max
            opt_u_v[0] = -my_sign(spat_deriv[3]) * self.u_v_max

        return (opt_u_theta[0], opt_u_v[0], in3, in4, in5)

    def opt_dstb(self, t, state, spat_deriv):
        opt_d_theta = hcl.scalar(0, "opt_d_theta")
        opt_d_v = hcl.scalar(0, "opt_d_v")
        # Just create and pass back, even though they're not used
        d3 = hcl.scalar(0, "d3")
        d4 = hcl.scalar(0, "d4")
        d5 = hcl.scalar(0, "d5")

        with hcl.if_(self.dMode == "max"):
            opt_d_theta[0] = my_sign(spat_deriv[2])*self.d_theta_max
            opt_d_v[0] = my_sign(spat_deriv[4])*self.d_v_max

        with hcl.if_(self.dMode == "min"):
            opt_d_theta[0] = -my_sign(spat_deriv[2])*self.d_theta_max
            opt_d_v[0] = -my_sign(spat_deriv[4])*self.d_v_max

        return (opt_d_theta[0], opt_d_v[0], d3, d4, d5) 

    def dynamics(self, t, state, uOpt, dOpt):
        x1_dot = hcl.scalar(0, "x_1_dot")
        x2_dot = hcl.scalar(0, "x_2_dot")
        x3_dot = hcl.scalar(0, "x_3_dot")
        x4_dot = hcl.scalar(0, "x_4_dot")
        x5_dot = hcl.scalar(0, "x_5_dot")

        x1_dot[0] = -state[3] + state[4] * hcl.cos(state[2]) + uOpt[0] * state[1]
        x2_dot[0] = state[4] * hcl.sin(state[2]) - uOpt[0] * state[0]
        x3_dot[0] = dOpt[0] - uOpt[0]
        x4_dot[0] = uOpt[1] 
        x5_dot[0] = dOpt[1]

        return (x1_dot[0], x2_dot[0], x3_dot[0], x4_dot[0], x5_dot[0])
