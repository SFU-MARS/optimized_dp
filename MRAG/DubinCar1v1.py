import heterocl as hcl
import numpy as np

""" 6D 1 vs. 1 DubinCar(3D) DYNAMICS IMPLEMENTATION 
 xA_dot = vA * cos(thetaA)
 yA_dot = vA * sin(thetaA)
 thetaA_dot = uA
 xD_dot = vD * cos(thetaD)
 yD_dot = vD * sin(thetaD)
 thetaD_dot = uD
 """

#TODO: Hanyang: remember to change the 6D computation graph!!!
class DubinsCar1v1:
    def __init__(self, x=[0, 0, 0, 0, 0, 0], uMin=-1, uMax=1, dMin=-1, 
                 dMax=1, uMode="min", dMode="max", speed_a=1.0, speed_d=1.5):
        self.x = x
        self.uMax = uMax
        self.uMin = uMin
        self.dMax = dMax
        self.dMin = dMin
        assert (uMode in ["min", "max"])
        self.uMode = uMode
        self.dMode = dMode
        self.speed_a = speed_a
        self.speed_d = speed_d     


    def dynamics(self, t, state, uOpt, dOpt):
        xA_dot = hcl.scalar(0, "xA_dot")
        yA_dot = hcl.scalar(0, "yA_dot")
        thetaA_dot = hcl.scalar(0, "thetaA_dot")
        xD_dot = hcl.scalar(0, "xD_dot")
        yD_dot = hcl.scalar(0, "yD_dot")
        thetaD_dot = hcl.scalar(0, "thetaD_dot")

        xA_dot[0] = self.speed_a*hcl.cos(state[2])
        yA_dot[0] = self.speed_a*hcl.sin(state[2])
        thetaA_dot[0] = uOpt[0]
        xD_dot[0] = self.speed_d*hcl.cos(state[5])
        yD_dot[0] = self.speed_d*hcl.sin(state[5])
        thetaD_dot[0] = dOpt[0]

        return (xA_dot[0], yA_dot[0], thetaA_dot[0], xD_dot[0], yD_dot[0], thetaD_dot[0])  


    def opt_ctrl(self, t, state, spat_deriv):
        opt_w = hcl.scalar(self.uMax, "opt_w")
        # Just create and pass back, even though they're not used
        in2 = hcl.scalar(0, "in2")
        in3 = hcl.scalar(0, "in3")
        in4 = hcl.scalar(0, "in4")

        with hcl.if_(spat_deriv[2] > 0):
            with hcl.if_(self.uMode == "min"):
                opt_w[0] = -opt_w
        with hcl.elif_(spat_deriv[2] < 0):
            with hcl.if_(self.uMode == "max"):
                opt_w[0] = -opt_w
                
        return (opt_w[0], in2[0], in3[0], in4[0])


    def opt_dstb(self, t, state, spat_deriv):
        """
        :param spat_deriv: tuple of spatial derivative in all dimensions
        :return: a tuple of optimal disturbances
        """
        opt_d = hcl.scalar(self.dMax, "opt_d")
        # Just create and pass back, even though they're not used
        d2 = hcl.scalar(0, "d2")
        d3 = hcl.scalar(0, "d3")
        d4 = hcl.scalar(0, "d4")
        
        with hcl.if_(spat_deriv[5] > 0):
            with hcl.if_(self.dMode == "min"):
                opt_d[0] = -opt_d
        with hcl.elif_(spat_deriv[5] < 0):
            with hcl.if_(self.dMode == "max"):
                opt_d[0] = -opt_d

        return (opt_d, d2[0], d3[0], d4[0])


    def optCtrl_inPython(self, spat_deriv):
        """
        :param spat_deriv: tuple of spatial derivative in all dimensions
        :return: a tuple of optimal control of the attacker
        """
        opt_u = self.uMax
        
        if spat_deriv[2] > 0:
            if self.uMode == "min":
                opt_u = -opt_u
        else:
            if self.uMode == "max":
                opt_u = -opt_u
    
        return opt_u
    
    
    def optDstb_inPython(self, spat_deriv):
        """
        :param spat_deriv: tuple of spatial derivative in all dimensions
        :return: a tuple of optimal disturbances
        """
        opt_d = self.dMax
        
        if spat_deriv[5] > 0:
            if self.dMode == "min":
                opt_d = -opt_d
        else:
            if self.dMode == "max":
                opt_d = -opt_d

        return opt_d
    
    
    def capture_set(self, grid, capture_radius, mode):
        xa, ya, xd, yd = np.meshgrid(grid.grid_points[0], grid.grid_points[1],
                                     grid.grid_points[2], grid.grid_points[3], indexing='ij')
        data = np.power(xa - xd, 2) + np.power(ya - yd, 2)
        if mode == "capture":
            return np.sqrt(data) - capture_radius
        if mode == "escape":
            return capture_radius - np.sqrt(data)
