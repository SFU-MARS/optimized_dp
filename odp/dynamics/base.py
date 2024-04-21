from abc import ABC, abstractmethod
import numpy as np

__all__ = ['DynamicsBase']

class DynamicsBase(ABC):

    def __init__(self, ctrl_range, dstb_range, mode='reach'):
        super().__init__()

        self.ctrl_range = np.asarray(ctrl_range)
        assert self.ctrl_range.shape[1] == self.ctrl_dims

        self.dstb_range = np.asarray(dstb_range)
        assert self.dstb_range.shape[1] == self.dstb_dims

        modes = {'reach': {"uMode": "min", "dMode": "max"},
                 'avoid': {"uMode": "max", "dMode": "min"}}
        self.mode = modes[mode]

    @property
    @abstractmethod
    def state_dims(self) -> int: pass

    @property
    @abstractmethod
    def ctrl_dims(self) -> int: pass

    @property
    @abstractmethod
    def dstb_dims(self) -> int: pass

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
