import numpy as np
from typing import List, Optional
from .firms import SwitchbackConfig

class SwitchbackExperiment:
    """Analyzes results from switchback price experiments"""
    
    def __init__(self, config: SwitchbackConfig):
        self.config = config
        self.price_history: List[float] = []
        self.demand_history: List[float] = []
        self.same_day_history: List[Optional[float]] = []
        
    def record_observation(self, 
                          price: float, 
                          demand: float,
                          same_day_demand: Optional[float] = None):
        """Record an observation from the experiment"""
        self.price_history.append(price)
        self.demand_history.append(demand)
        self.same_day_history.append(same_day_demand)
        
    def get_average_demand(self, price_level: float) -> float:
        """Get average demand for a given price level"""
        demands = [d for p, d in zip(self.price_history, self.demand_history) 
                  if np.isclose(p, price_level)]
        return np.mean(demands) if demands else 0
        
    def get_average_same_day_demand(self, price_level: float) -> float:
        """Get average same-day demand for a given price level"""
        demands = [d for p, d in zip(self.price_history, self.same_day_history) 
                  if np.isclose(p, price_level) and d is not None]
        return np.mean(demands) if demands else 0

    def estimate_demand_gradient(self, track_same_day: bool = False) -> float:
        """
        Estimate demand gradient using three-price estimator from Section 5
        Args:
            track_same_day: Whether to use same-day sales data
        Returns:
            Estimated demand gradient
        """
        if len(self.config.discount_levels) != 2:
            raise ValueError("This estimator requires exactly two discount levels")

        # Get price levels
        p0 = self.config.reference_price
        p1 = p0 - self.config.discount_levels[0]
        p2 = p0 - self.config.discount_levels[1]
        q0, q1, q2 = self.config.probabilities

        if track_same_day:
            # Use same-day sales based estimator
            N0 = self.get_average_same_day_demand(p0)
            N1 = self.get_average_same_day_demand(p1) 
            N2 = self.get_average_same_day_demand(p2)
            
            epsilon = self.config.discount_levels[0]
            return (N1 - N2 - (q2/q1)*(N0 - N1 - (N1 - N2))) / epsilon
        
        else:
            # Use total sales based estimator
            C0 = self.get_average_demand(p0)
            C1 = self.get_average_demand(p1)
            C2 = self.get_average_demand(p2)
            
            epsilon = self.config.discount_levels[0]
            scaling = q2 * (1 + q2/q1)
            return scaling * (2*C1 - C0 - C2) / epsilon