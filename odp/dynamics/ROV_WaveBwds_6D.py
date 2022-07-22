import heterocl as hcl
import numpy as np
import math
from odp.computeGraphs.CustomGraphFunctions import my_abs

class ROV_WaveBwds_6D:
    def __init__(self, x=[0, 0, 0, 0, 0, 0], uMin=np.array([-6, -15]), uMax=np.array([6, 15]), pMin=np.array([-0.2, -0.15]), pMax=np.array([0.2, 0.15]),
                 dMin=np.array([-0.02, -0.02, -0.01, -0.01]), dMax=np.array([0.02, 0.02, 0.01, 0.01]),
                 dims=6, uMode="min", dMode="max"):
        self.x = x
        self.uMode = uMode
        self.dMode = dMode

        # Object properties
        self.x    = x

        # Control bounds
        self.uMin = uMin
        self.uMax = uMax

        # Planner bounds
        self.pMin = pMin
        self.pMax = pMax

        # Disturbance bounds
        self.dMin = dMin
        self.dMax = dMax

        self.dims = dims

        # Define some constants
        self.m = 116    # Mass
        self.b = 116.2  # Displaced mass
        self.X_udot = -167.6 # Added mass x
        self.Z_wdot = -383  # Addeed mass z
        self.X_u    = 26.9  # linear damping x
        self.Z_w    = 0     # linear damping z
        self.X_uu   = 241.3     #quadratic damping x
        self.Z_ww   = 256.6     # quadratic damping z
        self.A      = 2     # wave amplitude
        self.omega  = 0.5*math.pi   # wave frequency
        self.g      = 9.81      # gravity
        self.k      = pow(self.omega,2)/self.g  # wavenumber
        self.W      = self.m * self.g   # weight of vehicle
        self.Buoy   = self.b * self.g    # buoyancy
        self.B      = np.array([[3.2,0.0], [0.6, 1.0]])

    def opt_ctrl(self, t, state, spat_deriv):
        uOpt1 = hcl.scalar(0, "uOpt1")
        uOpt2 = hcl.scalar(0, "uOpt2")
        uOpt3 = hcl.scalar(0, "uOpt3")
        uOpt4 = hcl.scalar(0, "uOpt4")

        parSum1 = hcl.scalar(0, "parSum1")
        parSum2 = hcl.scalar(0, "parSum2")

        with hcl.if_(self.uMode == "min"):
            with hcl.if_(spat_deriv[0] > 0):
                uOpt3[0] = -self.pMax[0]
            with hcl.elif_(spat_deriv[0] < 0):
                uOpt3[0] = -self.pMin[0]

            with hcl.if_(spat_deriv[1] > 0):
                uOpt4[0] = -self.pMax[1]
            with hcl.elif_(spat_deriv[1] < 0):
                uOpt4[0] = -self.pMin[1]

            parSum1[0] = (1/(self.m - self.X_udot)) * spat_deriv[2] * self.B[0,0] + (1/(self.m - self.Z_wdot)) * spat_deriv[3] * self.B[1,0]
            with hcl.if_(parSum1[0] > 0):
                uOpt1[0] = self.uMin[0]
            with hcl.elif_(parSum1[0] < 0):
                uOpt1[0] = self.uMax[0]

            parSum2[0] = (1/(self.m - self.X_udot)) * spat_deriv[2] * self.B[0,1] + (1/(self.m - self.Z_wdot)) * spat_deriv[3] * self.B[1,1]
            with hcl.if_(parSum2[0] > 0):
                uOpt2[0] = self.uMin[1]
            with hcl.elif_(parSum2[0] < 0):
                uOpt2[0] = self.uMax[1]

        return (uOpt1[0], uOpt2[0], uOpt3[0], uOpt4[0])

    def opt_dstb(self, t, state, spat_deriv):
        dOpt1 = hcl.scalar(0, "dOpt1")
        dOpt2 = hcl.scalar(0, "dOpt2")
        dOpt3 = hcl.scalar(0, "dOpt3")
        dOpt4 = hcl.scalar(0, "dOpt4")

        with hcl.if_(self.dMode == "max"):
            with hcl.if_(spat_deriv[0] > 0):
                dOpt1[0] = self.dMax[0]
            with hcl.elif_(spat_deriv[0] < 0):
                dOpt1[0] = self.dMin[0]

            with hcl.if_(spat_deriv[1] > 0):
                dOpt2[0] = self.dMax[1]
            with hcl.elif_(spat_deriv[1] < 0):
                dOpt2[0] = self.dMin[1]

            with hcl.if_(spat_deriv[2] > 0):
                dOpt3[0] = self.dMax[2]
            with hcl.elif_(spat_deriv[2] < 0):
                dOpt3[0] = self.dMin[2]

            with hcl.if_(spat_deriv[3] > 0):
                dOpt4[0] = self.dMax[3]
            with hcl.elif_(spat_deriv[3] < 0):
                dOpt4[0] = self.dMin[3]

        with hcl.elif_(self.dMode == "min"):
            with hcl.if_(spat_deriv[0] > 0):
                dOpt1[0] = self.dMin[0]
            with hcl.elif_(spat_deriv[0] < 0):
                dOpt1[0] = self.dMax[0]

            with hcl.if_(spat_deriv[1] > 0):
                dOpt2[0] = self.dMin[1]
            with hcl.elif_(spat_deriv[1] < 0):
                dOpt2[0] = self.dMax[1]

            with hcl.if_(spat_deriv[2] > 0):
                dOpt3[0] = self.dMin[2]
            with hcl.elif_(spat_deriv[2] < 0):
                dOpt3[0] = self.dMax[2]

            with hcl.if_(spat_deriv[3] > 0):
                dOpt4[0] = self.dMin[3]
            with hcl.elif_(spat_deriv[3] < 0):
                dOpt4[0] = self.dMax[3]

        return (dOpt1[0], dOpt2[0], dOpt3[0], dOpt4[0])

    def dynamics(self, t, state, uOpt, dOpt): # Assume order of state is (x_a, z_a, u_r, w_r, x, z)
        # Some constants for convenience
        G1 = hcl.scalar(0, "G1")
        F1 = hcl.scalar(0, "F1")
        sigma = hcl.scalar(0, "sigma")
        Phi_11   = hcl.scalar(0, "Phi_11")
        Phi_12   = hcl.scalar(0, "Phi_12")
        Phi_21   = hcl.scalar(0, "Phi_21")
        Phi_22   = hcl.scalar(0, "Phi_22")

        x1_dot = hcl.scalar(0, "x1_dot")
        x2_dot = hcl.scalar(0, "x2_dot")
        x3_dot = hcl.scalar(0, "x3_dot")
        x4_dot = hcl.scalar(0, "x4_dot")
        x5_dot = hcl.scalar(0, "x5_dot")
        x6_dot = hcl.scalar(0, "x6_dot")

        # Get the values
        G1[0] = self.A * self.omega * hcl.exp(-self.k * state[5])
        F1[0] = self.k * G1[0]
        sigma[0] = self.k * state[4] - self.omega * (-t[0])

        Phi_11[0] = F1[0] * (-hcl.sin(sigma[0]))
        Phi_12[0] = F1[0] * hcl.cos(sigma[0])
        Phi_21[0] =  Phi_12[0]
        Phi_22[0] = -Phi_11[0]

        # Question: What is B matrix
        x1_dot[0] = state[2] + G1[0] * hcl.cos(sigma[0]) + dOpt[0] - uOpt[2]
        x2_dot[0] = state[3] + G1[0] * (-hcl.sin(sigma[0])) + dOpt[1] - uOpt[3]

        x3_dot[0] = (1/(self.m - self.X_udot))*(-Phi_11[0]*(self.b - self.X_udot)* state[2] + (self.b - self.m)*(G1[0] * self.omega * hcl.sin(sigma[0])) +\
         Phi_11[0] * (state[2] + G1*hcl.cos(sigma[0])) + Phi_21[0]*(state[3] + G1[0]*(-hcl.sin(sigma[0]))) - (self.X_u + self.X_uu * my_abs(state[2]))*state[2] + \
        self.B[0,0]*uOpt[0] + self.B[0,1]*uOpt[1]) + dOpt[2]

        x4_dot[0] = (1/(self.m - self.Z_wdot))*(-Phi_22*(self.b - self.Z_wdot)*state[3] + \
         (self.b - self.m) * (G1 * self.omega * hcl.cos(sigma[0])) + \
         Phi_12[0] * (state[2] + G1 * hcl.cos(sigma[0])) + \
         Phi_22[0] * (state[3] + G1 * (-hcl.sin(sigma[0]))) - \
         (-(self.W - self.Buoy)) - (self.Z_w + self.Z_ww * my_abs(state[3])) * state[3] + \
         self.B[1,0] * uOpt[0] + self.B[1,1] * uOpt[1]) + \
         dOpt[3]

        x5_dot[0] = state[2] + G1[0] * hcl.cos(sigma[0]) + dOpt[0]
        x6_dot[0] = state[3] + G1[0] * (-hcl.sin(sigma[0])) + dOpt[1]
        return (x1_dot[0], x2_dot[0], x3_dot[0], x4_dot[0], x5_dot[0], x6_dot[0])

    def Hamiltonian(self, t_deriv, spatial_deriv):
        return t_deriv[0] * spatial_deriv[0] + t_deriv[1] * spatial_deriv[1] + t_deriv[2] * spatial_deriv[2] + t_deriv[3] * spatial_deriv[3] \
                + t_deriv[4] * spatial_deriv[4] + t_deriv[5] * spatial_deriv[5]
