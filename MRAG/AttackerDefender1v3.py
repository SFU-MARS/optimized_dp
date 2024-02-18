import heterocl as hcl
import math
import numpy as np

""" 8D 1 vs. 3 AttackerDefender DYNAMICS IMPLEMENTATION 
 xA11_dot = vA * u11
 xA12_dot = vA * u12
 xD11_dot = vD * d11
 xD12_dot = vD * d12
 xD21_dot = vD * d21
 xD22_dot = vD * d22
 xD31_dot = vD * d31
 xD32_dot = vD * d32
 """


class AttackerDefender1v3:
    def __init__(self, x=[0, 0, 0, 0, 0, 0, 0, 0], uMin=-1, uMax=1, dMin=-1,
                 dMax=1, uMode="min", dMode="max", speed_a=1.0, speed_d=1.5):
        """Creates 1 Attacker and 3 Defenders 
           The controls are the control inputs of the Attackers.
           The disturbances are the control inputs of the Defender.
        Args:
            x (list, optional): Initial state . Defaults to [0,0,0,0,0,0].
            uMin (list, optional): Lowerbound of user control. Defaults to [-1,-1].
            uMax (list, optional): Upperbound of user control. Defaults to [1,1].
            dMin (list, optional): Lowerbound of disturbance to user control, . Defaults to [-0.25,-0.25].
            dMax (list, optional): Upperbound of disturbance to user control. Defaults to [0.25,0.25].
            uMode (str, optional): Accepts either "min" or "max".
                                   * "min" : have optimal control reach goal
                                   * "max" : have optimal control avoid goal
                                   Defaults to "min".
            dMode (str, optional): Accepts whether "min" or "max" and should be opposite of uMode.
                                   Defaults to "max". reach-avoid game is a min(u) max(d) game
            speed_a (int, optional): The maximum speed of the attacker. Defaults to 1.0.
            speed_d (int, optional): The maximum speed of the defender. Defaults to 1.0.
        """
        self.x = x
        self.uMax = uMax
        self.uMin = uMin
        self.dMax = dMax
        self.dMin = dMin
        assert (uMode in ["min", "max"])
        self.uMode = uMode
        # if uMode == "min":
        #     assert (dMode == "max")
        # else:
        #     assert (dMode == "min")
        self.dMode = dMode
        # maximum speed of attackers and defenders
        self.speed_a = speed_a
        self.speed_d = speed_d

    def dynamics(self, t, state, uOpt, dOpt):
        # maximum velocity
        # vA = hcl.scalar(1.0, "vA")
        # vD = hcl.scalar(1.0, "vD")

        xA11_dot = hcl.scalar(0, "xA11_dot")
        xA12_dot = hcl.scalar(0, "xA12_dot")
        xD11_dot = hcl.scalar(0, "xD11_dot")
        xD12_dot = hcl.scalar(0, "xD12_dot")
        xD21_dot = hcl.scalar(0, "xD21_dot")
        xD22_dot = hcl.scalar(0, "xD22_dot")
        xD31_dot = hcl.scalar(0, "xD31_dot")
        xD32_dot = hcl.scalar(0, "xD32_dot")

        xA11_dot[0] = self.speed_a * uOpt[0]
        xA12_dot[0] = self.speed_a * uOpt[1]
        xD11_dot[0] = self.speed_d * dOpt[0]
        xD12_dot[0] = self.speed_d * dOpt[1]
        xD21_dot[0] = self.speed_d * dOpt[2]
        xD22_dot[0] = self.speed_d * dOpt[3]
        xD31_dot[0] = self.speed_d * dOpt[4]
        xD32_dot[0] = self.speed_d * dOpt[5]

        return xA11_dot[0], xA12_dot[0], xD11_dot[0], xD12_dot[0], xD21_dot[0], xD22_dot[0], xD31_dot[0], xD32_dot[0]

    def opt_ctrl(self, t, state, spat_deriv):
        """
        :param t: time t
        :param state: tuple of coordinates
        :param spat_deriv: tuple of spatial derivative in all dimensions
        :return:
        """
        # In 1v3 AttackerDefender, a(t) = [a1, a2]^T
        opt_a1 = hcl.scalar(0, "opt_a1")
        opt_a2 = hcl.scalar(0, "opt_a2")      
        # Just create and pass back, even though they're not used
        in3 = hcl.scalar(0, "in3")
        in4 = hcl.scalar(0, "in4")
        in5 = hcl.scalar(0, "in5")
        in6 = hcl.scalar(0, "in6")
        in7 = hcl.scalar(0, "in7")
        in8 = hcl.scalar(0, "in8")
        # declare the hcl scalars for relevant spat_derivs
        deriv1 = hcl.scalar(0, "deriv1")
        deriv2 = hcl.scalar(0, "deriv2")

        deriv1[0] = spat_deriv[0]
        deriv2[0] = spat_deriv[1]

        ctrl_len1 = hcl.sqrt(deriv1[0] * deriv1[0] + deriv2[0] * deriv2[0])     

        if self.uMode == "min":
            with hcl.if_(ctrl_len1 == 0):
                opt_a1[0] = 0.0
                opt_a2[0] = 0.0
            with hcl.else_():
                opt_a1[0] = -deriv1[0] / ctrl_len1
                opt_a2[0] = -deriv2[0] / ctrl_len1

        else:
            with hcl.if_(ctrl_len1 == 0):
                opt_a1[0] = 0.0
                opt_a2[0] = 0.0
            with hcl.else_():
                opt_a1[0] = deriv1[0] / ctrl_len1
                opt_a2[0] = deriv2[0] / ctrl_len1
            
        return opt_a1[0], opt_a2[0], in3[0], in4[0], in5[0], in6[0], in7[0], in8[0]

    def opt_dstb(self, t, state, spat_deriv):
        """
        :param spat_deriv: tuple of spatial derivative in all dimensions
        :return: a tuple of optimal disturbances
        """
        # Graph takes in 4 possible inputs, by default, for now
        d1 = hcl.scalar(0, "d1")
        d2 = hcl.scalar(0, "d2")
        d3 = hcl.scalar(0, "d3")
        d4 = hcl.scalar(0, "d4")
        d5 = hcl.scalar(0, "d5")
        d6 = hcl.scalar(0, "d6")
        # Just create and pass back, even though they're not used
        d7 = hcl.scalar(0, "d7")
        d8 = hcl.scalar(0, "d8")

        deriv1 = hcl.scalar(0, "deriv1")
        deriv2 = hcl.scalar(0, "deriv2")
        deriv3 = hcl.scalar(0, "deriv3")
        deriv4 = hcl.scalar(0, "deriv4")
        deriv5 = hcl.scalar(0, "deriv5")
        deriv6 = hcl.scalar(0, "deriv6")
        deriv1[0] = spat_deriv[2]
        deriv2[0] = spat_deriv[3]
        deriv3[0] = spat_deriv[4]
        deriv4[0] = spat_deriv[5]
        deriv5[0] = spat_deriv[6]
        deriv6[0] = spat_deriv[7]

        dstb_len1 = hcl.sqrt(deriv1[0] * deriv1[0] + deriv2[0] * deriv2[0])
        dstb_len2 = hcl.sqrt(deriv3[0] * deriv3[0] + deriv4[0] * deriv4[0])
        dstb_len3 = hcl.sqrt(deriv5[0] * deriv5[0] + deriv6[0] * deriv6[0])

        # with hcl.if_(self.dMode == "max"):
        if self.dMode == 'max':
            with hcl.if_(dstb_len1 == 0):
                d1[0] = 0.0
                d2[0] = 0.0
            with hcl.else_():
                d1[0] = deriv1[0] / dstb_len1
                d2[0] = deriv2[0] / dstb_len1
            
            with hcl.if_(dstb_len2 == 0):
                d3[0] = 0.0
                d4[0] = 0.0
            with hcl.else_():
                d3[0] = deriv3[0] / dstb_len2
                d4[0] = deriv4[0] / dstb_len2

            with hcl.if_(dstb_len3 == 0):
                d5[0] = 0.0
                d6[0] = 0.0
            with hcl.else_():
                d5[0] = deriv5[0] / dstb_len3
                d6[0] = deriv6[0] / dstb_len3
        else:
            with hcl.if_(dstb_len1 == 0):
                d1[0] = 0.0
                d2[0] = 0.0
            with hcl.else_():
                d1[0] = -deriv1[0]/ dstb_len1
                d2[0] = -deriv2[0] / dstb_len1
            
            with hcl.if_(dstb_len2 == 0):
                d3[0] = 0.0
                d4[0] = 0.0
            with hcl.else_():
                d3[0] = -deriv3[0]/ dstb_len2
                d4[0] = -deriv4[0] / dstb_len2
            
            with hcl.if_(dstb_len3 == 0):
                d5[0] = 0.0
                d6[0] = 0.0
            with hcl.else_():
                d5[0] = -deriv5[0]/ dstb_len3
                d6[0] = -deriv6[0] / dstb_len3

        return d1[0], d2[0], d3[0], d4[0], d5[0], d6[0], d7[0], d8[0]

        # The below function can have whatever form or parameters users want
        # These functions are not used in HeteroCL program, hence is pure Python code and
        # can be used after the value function has been obtained.

    def optCtrl_inPython(self, spat_deriv):
        """
        :param spat_deriv: tuple of spatial derivative in all dimensions
        :return: a tuple of optimal control of two attackers
        """
        opt_a1 = self.uMax
        opt_a2 = self.uMax
        deriv1 = spat_deriv[0]
        deriv2 = spat_deriv[1]
        ctrl_len1 = np.sqrt(deriv1*deriv1 + deriv2*deriv2)

        # The initialized control only change sign in the following cases
        if self.uMode == "min":
            if ctrl_len1 == 0:
                opt_a1 = 0.0
                opt_a2 = 0.0
            else:
                opt_a1 = -deriv1 / ctrl_len1
                opt_a2 = -deriv2 / ctrl_len1
           
        else:
            if ctrl_len1 == 0:
                opt_a1 = 0.0
                opt_a2 = 0.0
            else:
                opt_a1 = deriv1 / ctrl_len1
                opt_a2 = deriv2 / ctrl_len1
            
        return (opt_a1, opt_a2)
    
    def optDstb_inPython(self, spat_deriv):
        """
        :param spat_deriv: tuple of spatial derivative in all dimensions
        :return: a tuple of optimal control of the defender (disturbances)
        """
        opt_d1 = self.dMax
        opt_d2 = self.dMax
        opt_d3 = self.dMax
        opt_d4 = self.dMax
        opt_d5 = self.dMax
        opt_d6 = self.dMax

        deriv3 = spat_deriv[2]
        deriv4 = spat_deriv[3]
        deriv5 = spat_deriv[4]
        deriv6 = spat_deriv[5]
        deriv7 = spat_deriv[6]
        deriv8 = spat_deriv[7]

        dstb_len1 = np.sqrt(deriv3*deriv3 + deriv4*deriv4)
        dstb_len2 = np.sqrt(deriv5*deriv5 + deriv6*deriv6)
        dstb_len3 = np.sqrt(deriv7*deriv7 + deriv8*deriv8)

        # The initialized control only change sign in the following cases
        if self.dMode == "max":
            if dstb_len1 == 0:
                opt_d1 = 0.0
                opt_d2 = 0.0
            else:
                opt_d1 = self.speed_d*deriv3 / dstb_len1
                opt_d2 = self.speed_d*deriv4 / dstb_len1
            
            if dstb_len2 == 0:
                opt_d3 = 0.0
                opt_d4 = 0.0
            else:
                opt_d3 = self.speed_d*deriv5 / dstb_len2
                opt_d4 = self.speed_d*deriv6 / dstb_len2
            
            if dstb_len3 == 0:
                opt_d5 = 0.0
                opt_d6 = 0.0
            else:
                opt_d5 = self.speed_d*deriv7 / dstb_len3
                opt_d6 = self.speed_d*deriv8 / dstb_len3

        else:
            if dstb_len1 == 0:
                opt_d1 = 0.0
                opt_d2 = 0.0
            else:
                opt_d1 = -self.speed_d*deriv3 / dstb_len1
                opt_d2 = -self.speed_d*deriv4 / dstb_len1
            
            if dstb_len2 == 0:
                opt_d3 = 0.0
                opt_d4 = 0.0
            else:
                opt_d3 = -self.speed_d*deriv5 / dstb_len2
                opt_d4 = -self.speed_d*deriv6 / dstb_len2
            
            if dstb_len3 == 0:
                opt_d5 = 0.0
                opt_d6 = 0.0
            else:
                opt_d5 = -self.speed_d*deriv7 / dstb_len3
                opt_d6 = -self.speed_d*deriv8 / dstb_len3

        return (opt_d1, opt_d2, opt_d3, opt_d4, opt_d5, opt_d6)

    def capture_set1(self, grid, capture_radius, mode):
        data = np.power(grid.vs[0] - grid.vs[2], 2) + np.power(grid.vs[1] -grid.vs[3], 2)
        if mode == "capture":
            return np.sqrt(data) - capture_radius
        if mode == "escape":
            return capture_radius - np.sqrt(data)

    def capture_set2(self, grid, capture_radius, mode):
        data = np.power(grid.vs[0] - grid.vs[4], 2) + np.power(grid.vs[1] -grid.vs[5], 2)
        if mode == "capture":
            return np.sqrt(data) - capture_radius
        if mode == "escape":
            return capture_radius - np.sqrt(data)
    
    def capture_set3(self, grid, capture_radius, mode):
        data = np.power(grid.vs[0] - grid.vs[6], 2) + np.power(grid.vs[1] -grid.vs[7], 2)
        if mode == "capture":
            return np.sqrt(data) - capture_radius
        if mode == "escape":
            return capture_radius - np.sqrt(data)
