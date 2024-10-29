from abc import ABC, abstractmethod
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
        # No lower bound - can be negative if price > value
        return self.value - price if purchased else 0.0
        
    def marginal_utility(self, consumption: float) -> float:
        """Marginal utility is value for first unit, 0 after"""
        return self.value if consumption < 1 else 0.0
        
    def inverse_marginal_utility(self, mu: float) -> float:
        """Inverse is 1 if mu >= value, 0 otherwise"""
        return 1.0 if mu >= self.value else 0.0

class CRRA(UtilityFunction):
    """CRRA utility for unit demand"""
    def __init__(self, gamma: float):
        if gamma == 1:
            raise ValueError("gamma=1 is log utility case, use different class")
        self.gamma = gamma
    
    def utility(self, purchased: bool, price: float) -> float:
        """
        For unit demand:
        U = (1^(1-gamma))/(1-gamma) - p if purchased, 0 otherwise
        Simplifies to U = 1/(1-gamma) - p if purchased
        """
        if purchased:
            return 1/(1-self.gamma) - price
        return 0.0

    def marginal_utility(self, consumption: float) -> float:
        """
        For unit demand, marginal utility is just 1 for the first unit
        """
        return 1.0 if consumption < 1 else 0.0
        
    def inverse_marginal_utility(self, mu: float) -> float:
        """
        For unit demand, returns 1 if mu >= 1, 0 otherwise
        """
        return 1.0 if mu >= 1 else 0.0
    
class QuasiLinear(UtilityFunction):
    """Quasilinear utility"""
    def __init__(self, alpha: float):
        self.alpha = alpha
    
    def utility(self, consumption: float, price: float) -> float:
        """U = alpha*log(1 + c) - p"""
        # Allow negative utility from price payment
        return self.alpha * np.log(1 + consumption) - price

    def marginal_utility(self, consumption: float) -> float:
        """U'(c) = alpha/(1+c)"""
        return self.alpha/(1 + consumption)
        
    def inverse_marginal_utility(self, mu: float) -> float:
        """(U')^(-1)(mu) = alpha/mu - 1"""
        return self.alpha/mu - 1