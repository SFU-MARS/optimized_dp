from abc import ABC, abstractmethod

class Model(ABC):

    state_dims: int
    ctrl_dims: int
    dstb_dims: int

    @abstractmethod
    def opt_dstb(self, d, dv, t, x): 
        """
        Update optimal disturbance.

        Args:
            d (hcl.tensor): optimal disturbance to be updated
            t (hcl.tensor): 2-element tensor with current time step and next.
            x (hcl.tensor): state 
            dv (hcl.tensor): spatial derivative (dV/dx)

        Returns: None
        """
        pass

    @abstractmethod
    def opt_ctrl(self, u, dv, t, x): 
        """
        Update optimal control.

        Args:
            u (hcl.tensor): optimal control to be updated
            t (hcl.tensor): 2-element tensor with current time step and next.
            x (hcl.tensor): state 
            dv (hcl.tensor): spatial derivative (dV/dx)

        Returns: None
        """
        pass

    @abstractmethod 
    def dynamics(self, dx, t, x, u, d):
        """
        Update state change.

        Similar to opt_dstb.

        Args:
            dx (hcl.tensor): state derivative (dx/dt) to be updated
            t (hcl.tensor): 2-element tensor with current time step and next.
            x (hcl.tensor): state 
            u (hcl.tensor): control input
            d (hcl.tensor): disturbance

        Returns: None
        """
        pass
