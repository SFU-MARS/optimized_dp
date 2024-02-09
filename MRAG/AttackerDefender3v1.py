import heterocl as hcl
import math
import numpy as np

""" 8D 3v1 AttackerDefender DYNAMICS IMPLEMENTATION 
 xA11_dot = vA * u11
 xA12_dot = vA * u12
 xA21_dot = vA * u21
 xA22_dot = vA * u22
 xA31_dot = vA * u31
 xA32_dot = vA * u32
 xD1_dot = vD * d1
 xD2_dot = vD * d2
 """


class AttackerDefender3v1:
    #TODO: need to implement the dynamics function, 2nd, Feb, 2024.
    def __init__(self, x=[0, 0, 0, 0, 0, 0], uMin=-1, uMax=1, dMin=-1,
                 dMax=1, uMode="min", dMode="max", speed_a=1.0, speed_d=1.5):
        """Creates 3 Attackers and 1 Defender with the following states:
           X1 position, Y1 position, X2 position, Y2 position, X3 position, Y3 position
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
        xA21_dot = hcl.scalar(0, "xA21_dot")
        xA22_dot = hcl.scalar(0, "xA22_dot")
        xA31_dot = hcl.scalar(0, "xA31_dot")
        xA32_dot = hcl.scalar(0, "xA32_dot")
        xD1_dot = hcl.scalar(0, "xD1_dot")
        xD2_dot = hcl.scalar(0, "xD2_dot")

        xA11_dot[0] = self.speed_a * uOpt[0]
        xA12_dot[0] = self.speed_a * uOpt[1]
        xA21_dot[0] = self.speed_a * uOpt[2] 
        xA22_dot[0] = self.speed_a * uOpt[3] 
        xA31_dot[0] = self.speed_a * uOpt[4] 
        xA32_dot[0] = self.speed_a * uOpt[5] 
        xD1_dot[0] = self.speed_d * dOpt[0]
        xD2_dot[0] = self.speed_d * dOpt[1]

        return xA11_dot[0], xA12_dot[0], xA21_dot[0], xA22_dot[0], xA31_dot[0], xA32_dot[0], xD1_dot[0], xD2_dot[0]

    def opt_ctrl(self, t, state, spat_deriv):
        """
        :param t: time t
        :param state: tuple of coordinates
        :param spat_deriv: tuple of spatial derivative in all dimensions
        :return:
        """
        # In 3v1AttackerDefender, a(t) = [a1, a2, a3, a4]^T
        opt_a1 = hcl.scalar(0, "opt_a1")
        opt_a2 = hcl.scalar(0, "opt_a2")
        opt_a3 = hcl.scalar(0, "opt_a3")
        opt_a4 = hcl.scalar(0, "opt_a4")  
        opt_a5 = hcl.scalar(0, "opt_a5")
        opt_a6 = hcl.scalar(0, "opt_a6")       
        # Just create and pass back, even though they're not used
        # in3 = hcl.scalar(0, "in3")
        # in4 = hcl.scalar(0, "in4")
        # declare the hcl scalars for relevant spat_derivs
        deriv1 = hcl.scalar(0, "deriv1")
        deriv2 = hcl.scalar(0, "deriv2")
        deriv3 = hcl.scalar(0, "deriv3")
        deriv4= hcl.scalar(0, "deriv4")
        deriv5 = hcl.scalar(0, "deriv5")
        deriv6= hcl.scalar(0, "deriv6")
        deriv1[0] = spat_deriv[0]
        deriv2[0] = spat_deriv[1]
        deriv3[0] = spat_deriv[2]
        deriv4[0] = spat_deriv[3]
        deriv5[0] = spat_deriv[4]
        deriv6[0] = spat_deriv[5]
        ctrl_len1 = hcl.sqrt(deriv1[0] * deriv1[0] + deriv2[0] * deriv2[0])     
        ctrl_len2 = hcl.sqrt(deriv3[0] * deriv3[0] + deriv4[0] * deriv4[0])
        ctrl_len3 = hcl.sqrt(deriv5[0] * deriv5[0] + deriv6[0] * deriv6[0])
        if self.uMode == "min":
            with hcl.if_(ctrl_len1 == 0):
                opt_a1[0] = 0.0
                opt_a2[0] = 0.0
            with hcl.else_():
                opt_a1[0] = -deriv1[0] / ctrl_len1
                opt_a2[0] = -deriv2[0] / ctrl_len1
            with hcl.if_(ctrl_len2 == 0):
                opt_a3[0] = 0.0
                opt_a4[0] = 0.0
            with hcl.else_():
                opt_a3[0] = -deriv3[0] / ctrl_len2
                opt_a4[0] = -deriv4[0] / ctrl_len2
            with hcl.if_(ctrl_len3 == 0):
                opt_a5[0] = 0.0
                opt_a6[0] = 0.0
            with hcl.else_():
                opt_a5[0] = -deriv5[0] / ctrl_len3
                opt_a6[0] = -deriv6[0] / ctrl_len3
        else:
            with hcl.if_(ctrl_len1 == 0):
                opt_a1[0] = 0.0
                opt_a2[0] = 0.0
            with hcl.else_():
                opt_a1[0] = deriv1[0] / ctrl_len1
                opt_a2[0] = deriv2[0] / ctrl_len1
            with hcl.if_(ctrl_len2 == 0):
                opt_a3[0] = 0.0
                opt_a4[0] = 0.0
            with hcl.else_():
                opt_a3[0] = deriv3[0] / ctrl_len2
                opt_a4[0] = deriv4[0] / ctrl_len2
            with hcl.if_(ctrl_len3 == 0):
                opt_a5[0] = 0.0
                opt_a6[0] = 0.0
            with hcl.else_():
                opt_a5[0] = deriv5[0] / ctrl_len3
                opt_a6[0] = deriv6[0] / ctrl_len3
        # return 3, 4 even if you don't use them
        return opt_a1[0], opt_a2[0], opt_a3[0], opt_a4[0], opt_a5[0], opt_a6[0]

    def opt_dstb(self, t, state, spat_deriv):
        """
        :param spat_deriv: tuple of spatial derivative in all dimensions
        :return: a tuple of optimal disturbances
        """
        # Graph takes in 4 possible inputs, by default, for now
        d1 = hcl.scalar(0, "d1")
        d2 = hcl.scalar(0, "d2")
        # Just create and pass back, even though they're not used
        d3 = hcl.scalar(0, "d3")
        d4 = hcl.scalar(0, "d4")
        # the same procedure in opt_ctrl
        deriv1 = hcl.scalar(0, "deriv1")
        deriv2 = hcl.scalar(0, "deriv2")
        deriv1[0] = spat_deriv[6]
        deriv2[0] = spat_deriv[7]
        dstb_len = hcl.sqrt(deriv1[0] * deriv1[0] + deriv2[0] * deriv2[0])
        # with hcl.if_(self.dMode == "max"):
        if self.dMode == 'max':
            with hcl.if_(dstb_len == 0):
                d1[0] = 0.0
                d2[0] = 0.0
            with hcl.else_():
                d1[0] = deriv1[0] / dstb_len
                d2[0] = deriv2[0] / dstb_len
        else:
            with hcl.if_(dstb_len == 0):
                d1[0] = 0.0
                d2[0] = 0.0
            with hcl.else_():
                d1[0] = -deriv1[0]/ dstb_len
                d2[0] = -deriv2[0] / dstb_len

        return d1[0], d2[0], d3[0], d4[0]

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
        opt_a3 = self.uMax
        opt_a4 = self.uMax
        opt_a5 = self.uMax
        opt_a6 = self.uMax
        deriv1 = spat_deriv[0]
        deriv2 = spat_deriv[1]
        deriv3 = spat_deriv[2]
        deriv4 = spat_deriv[3]
        deriv5 = spat_deriv[4]
        deriv6 = spat_deriv[5]
        ctrl_len1 = np.sqrt(deriv1*deriv1 + deriv2*deriv2)
        ctrl_len2 = np.sqrt(deriv3*deriv3 + deriv4*deriv4)
        ctrl_len3 = np.sqrt(deriv5*deriv5 + deriv6*deriv6)
        # The initialized control only change sign in the following cases
        if self.uMode == "min":
            if ctrl_len1 == 0:
                opt_a1 = 0.0
                opt_a2 = 0.0
            else:
                opt_a1 = -deriv1 / ctrl_len1
                opt_a2 = -deriv2 / ctrl_len1
                
            if ctrl_len2 == 0:
                opt_a3 = 0.0 
                opt_a4 = 0.0
            else:
                opt_a3 = -deriv3 / ctrl_len2
                opt_a4 = -deriv4 / ctrl_len2
                
            if ctrl_len3 == 0:
                opt_a5 = 0.0 
                opt_a6 = 0.0
            else:
                opt_a5 = -deriv5 / ctrl_len3
                opt_a6 = -deriv6 / ctrl_len3
        else:
            if ctrl_len1 == 0:
                opt_a1 = 0.0
                opt_a2 = 0.0
            else:
                opt_a1 = deriv1 / ctrl_len1
                opt_a2 = deriv2 / ctrl_len1
                
            if ctrl_len2 == 0:
                opt_a3 = 0.0 
                opt_a4 = 0.0
            else:
                opt_a3 = deriv3 / ctrl_len2
                opt_a4 = deriv4 / ctrl_len2
            
            if ctrl_len3 == 0:
                opt_a5 = 0.0 
                opt_a6 = 0.0
            else:
                opt_a5 = deriv5 / ctrl_len3
                opt_a6 = deriv6 / ctrl_len3
                
        return (opt_a1, opt_a2, opt_a3, opt_a4, opt_a5, opt_a6)
    
    def optDstb_inPython(self, spat_deriv):
        """
        :param spat_deriv: tuple of spatial derivative in all dimensions
        :return: a tuple of optimal control of the defender (disturbances)
        """
        opt_d1 = self.dMax
        opt_d2 = self.dMax
        deriv5 = spat_deriv[6]
        deriv6 = spat_deriv[7]
        dstb_len = np.sqrt(deriv5*deriv5 + deriv6*deriv6)
        # The initialized control only change sign in the following cases
        if self.dMode == "max":
            if dstb_len == 0:
                opt_d1 = 0.0
                opt_d2 = 0.0
            else:
                opt_d1 = self.speed_d*deriv5 / dstb_len
                opt_d2 = self.speed_d*deriv6 / dstb_len
        else:
            if dstb_len == 0:
                opt_d1 = 0.0
                opt_d2 = 0.0
            else:
                opt_d1 = -self.speed_d*deriv5 / dstb_len
                opt_d2 = -self.speed_d*deriv6 / dstb_len
        return (opt_d1, opt_d2)

    def capture_set1(self, grid, capture_radius, mode):
        ## todo: not sure whether this works or not
        # Meshgrid is too expensive for 6D. So gotta be more cheap with our memory usage
        # xa1, ya1, xa2, ya2, xd, yd = np.meshgrid(grid.grid_points[0], grid.grid_points[1],
        #                              grid.grid_points[2], grid.grid_points[3],
        #                              grid.grid_points[4], grid.grid_points[5], indexing='ij')
        # data = np.power(xa1 - xd, 2) + np.power(ya1 - yd, 2)
        # if mode == "capture":
        #     return np.sqrt(data) - capture_radius
        # if mode == "escape":
        #     return capture_radius - np.sqrt(data)

        data = np.power(grid.vs[0] - grid.vs[6], 2) + np.power(grid.vs[1] -grid.vs[7], 2)
        if mode == "capture":
            return np.sqrt(data) - capture_radius
        if mode == "escape":
            return capture_radius - np.sqrt(data)


        # this function is the distance between 1 attacker and 1 defender
        # data = np.zeros(grid.pts_each_dim)
        #
        # data = data + np.power(grid.vs[0] - grid.vs[2], 2)
        # data = data + np.power(grid.vs[1] - grid.vs[3], 2)
        # # data = np.sqrt(data) - radius
        # if mode == "capture":
        #     return np.sqrt(data) - capture_radius
        # if mode == "escape":
        #     return capture_radius - np.sqrt(data)

    def capture_set2(self, grid, capture_radius, mode):
        data = np.power(grid.vs[2] - grid.vs[6], 2) + np.power(grid.vs[3] -grid.vs[7], 2)
        if mode == "capture":
            return np.sqrt(data) - capture_radius
        if mode == "escape":
            return capture_radius - np.sqrt(data)
    
    def capture_set3(self, grid, capture_radius, mode):
        data = np.power(grid.vs[4] - grid.vs[6], 2) + np.power(grid.vs[5] -grid.vs[7], 2)
        if mode == "capture":
            return np.sqrt(data) - capture_radius
        if mode == "escape":
            return capture_radius - np.sqrt(data)
