from abc import abstractmethod, ABC


class DynamicalSystem(ABC):
    @abstractmethod
    def opt_ctrl(self):
        pass

    @abstractmethod
    def opt_dstb(self):
        pass

    @abstractmethod
    def dynamics(self):
        pass
