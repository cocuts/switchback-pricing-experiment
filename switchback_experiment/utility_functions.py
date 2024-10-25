from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

class UtilityFunction(ABC):
    """Abstract base class for utility functions"""
    
    @abstractmethod
    def utility(self, consumption: float, *args, **kwargs) -> float:
        """Compute utility"""
        pass
        
    @abstractmethod
    def marginal_utility(self, consumption: float, *args, **kwargs) -> float:
        """Compute marginal utility"""
        pass
    
    @abstractmethod
    def inverse_marginal_utility(self, mu: float, *args, **kwargs) -> float:
        """Compute inverse of marginal utility"""
        pass

class LinearUnitDemand(UtilityFunction):
    """Linear utility with unit demand"""
    def __init__(self, value: float):
        self.value = value
    
    def utility(self, purchased: bool, price: float) -> float:
        """U = v - p if purchased, 0 otherwise"""
        if purchased:
            return self.value - price
        return 0.0
        
    def marginal_utility(self, consumption: float) -> float:
        """Marginal utility is value for first unit, 0 after"""
        if consumption < 1:
            return self.value
        return 0.0
        
    def inverse_marginal_utility(self, mu: float) -> float:
        """Inverse is 1 if mu >= value, 0 otherwise"""
        return 1.0 if mu >= self.value else 0.0

class CRRA(UtilityFunction):
    """CRRA utility"""
    def __init__(self, gamma: float):
        if gamma == 1:
            raise ValueError("gamma=1 is log utility case, use different class")
        self.gamma = gamma
    
    def utility(self, consumption: float, price: float) -> float:
        """U = (c^(1-gamma))/(1-gamma) - p"""
        if consumption <= 0:
            return float('-inf')
        return (consumption**(1-self.gamma))/(1-self.gamma) - price

    def marginal_utility(self, consumption: float) -> float:
        """U'(c) = c^(-gamma)"""
        if consumption <= 0:
            return float('inf')
        return consumption**(-self.gamma)
        
    def inverse_marginal_utility(self, mu: float) -> float:
        """(U')^(-1)(mu) = mu^(-1/gamma)"""
        return mu**(-1/self.gamma)

class QuasiLinear(UtilityFunction):
    """Quasilinear utility"""
    def __init__(self, alpha: float):
        self.alpha = alpha
    
    def utility(self, consumption: float, price: float) -> float:
        """U = alpha*log(1 + c) - p"""
        if consumption <= 0:
            return 0.0
        return self.alpha * np.log(1 + consumption) - price

    def marginal_utility(self, consumption: float) -> float:
        """U'(c) = alpha/(1+c)"""
        if consumption <= 0:
            return self.alpha
        return self.alpha/(1 + consumption)
        
    def inverse_marginal_utility(self, mu: float) -> float:
        """(U')^(-1)(mu) = alpha/mu - 1"""
        return self.alpha/mu - 1 if mu > 0 else float('inf')