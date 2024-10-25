from typing import List, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar

from .utility_functions import UtilityFunction, LinearUnitDemand

@dataclass
class PriceInfo:
    """Information about the price distribution in the experiment"""
    prices: List[float]
    probabilities: List[float]
    delta: float
    post_experiment_price: Optional[float] = None

class ConsumerBase(ABC):
    """Abstract base class for forward-looking consumers"""
    
    def __init__(self, 
                 utility_function: UtilityFunction,
                 gamma: float,
                 budget_constraint: float = float('inf')):
        self.utility_function = utility_function
        self.gamma = gamma
        self.budget_constraint = budget_constraint
        self.has_purchased = False
    
    @abstractmethod
    def get_continuation_value(self, price: float, price_info: PriceInfo) -> float:
        """Get value of waiting given current state"""
        pass
    
    def should_purchase(self, price: float, price_info: PriceInfo) -> bool:
        """Determine whether to purchase based on immediate utility vs continuation value"""
        if self.has_purchased or price > self.budget_constraint:
            return False
            
        immediate_util = self.utility_function.utility(True, price)
        continuation_value = self.get_continuation_value(price, price_info)
        
        if immediate_util >= continuation_value:
            self.has_purchased = True
            return True
            
        return False

class AnalyticConsumer(ConsumerBase):
    """
    Implementation using asymptotic result (δ -> 0) for linear utility case
    Based on Corollary 3.2 from paper
    """
    
    def __init__(self, value: float, gamma: float, budget_constraint: float = float('inf')):
        util = LinearUnitDemand(value)
        super().__init__(util, gamma, budget_constraint)
    
    def get_continuation_value(self, price: float, price_info: PriceInfo) -> float:
        """Calculate option value from Corollary 3.2"""
        # Probability of seeing a lower price
        Q = sum(prob for p, prob in zip(price_info.prices, price_info.probabilities) 
                if p < price)
        
        if Q > 0:
            # Expected price discount when seeing lower price
            E = sum(prob * (price - p) for p, prob in zip(price_info.prices, price_info.probabilities)
                    if p < price) / Q
        else:
            E = 0
            
        return (self.gamma / (1 - self.gamma)) * Q * E

class GridpointConsumer(ConsumerBase):
    """
    General implementation using standard exogenous gridpoints method
    Can handle any utility function specification
    """
    
    def __init__(self, 
                 utility_function: UtilityFunction,
                 gamma: float,
                 grid_size: int = 100,
                 budget_constraint: float = float('inf')):
        super().__init__(utility_function, gamma, budget_constraint)
        self.grid_size = grid_size
        self.value_function = None
        self.decision_function = None
        self.m_grid = None
    
    def get_grid(self, price: float) -> np.ndarray:
        """Create exogenous grid based on price and budget"""
        max_m = min(price * 5000, self.budget_constraint)
        return np.linspace(0, max_m, self.grid_size)

    def euler_error(self, decision: bool, price: float, price_info: PriceInfo) -> float:
        """Calculate euler equation error at given decision"""
        if price > self.budget_constraint:
            return float('-inf')
            
        immediate_utility = self.utility_function.utility(decision, price)
        
        # Expected future value
        future_value = 0
        for p, prob in zip(price_info.prices, price_info.probabilities):
            if self.value_function is not None:
                future_value += prob * self.value_function(p)
            else:
                # Terminal period utility
                future_value += prob * self.utility_function.utility(True, p)
                
        return immediate_utility - self.gamma * future_value

    def solve_period(self, price: float, price_info: PriceInfo) -> Tuple[np.ndarray, np.ndarray]:
        """Solve one period's decision problem"""
        self.m_grid = self.get_grid(price)
        decisions = np.zeros_like(self.m_grid)
        
        for i, m in enumerate(self.m_grid):
            if m >= price:  # Can afford to purchase
                buy_utility = self.euler_error(True, price, price_info)
                wait_utility = self.euler_error(False, price, price_info)
                decisions[i] = 1 if buy_utility > wait_utility else 0
                
        return self.m_grid, decisions

    def update_value_function(self, decisions: np.ndarray, price: float):
        """Update value function interpolation"""
        v_points = np.array([
            self.utility_function.utility(bool(d), price) 
            for d in decisions
        ])
        
        self.value_function = interp1d(
            self.m_grid, v_points,
            bounds_error=False,
            fill_value=(v_points[0], v_points[-1])
        )
        
        self.decision_function = interp1d(
            self.m_grid, decisions,
            bounds_error=False,
            fill_value=(decisions[0], decisions[-1])
        )
    
    def get_continuation_value(self, price: float, price_info: PriceInfo) -> float:
        """Get continuation value using current value function"""
        if self.value_function is None:
            _, decisions = self.solve_period(price, price_info)
            self.update_value_function(decisions, price)
            
        return float(self.value_function(price))