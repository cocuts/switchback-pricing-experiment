from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict

class BaseConsumer:
    def __init__(self, value: float, gamma: float):
        self.value = value
        self.gamma = gamma  # Patience level
        self.has_purchased = False
    
    def utility(self, price: float) -> float:
        """Calculate immediate utility from purchasing at given price"""
        return max(0, self.value - price)
    
@dataclass
class PriceInfo:
    """Information about the price distribution in the experiment"""
    prices: List[float]  # List of possible prices
    probabilities: List[float]  # Corresponding probabilities
    delta: float  # Experiment ending probability
    post_experiment_price: Optional[float] = None  # Expected price after experiment

class ForwardLookingConsumerBase(BaseConsumer, ABC):
    """Abstract base class for forward-looking consumers"""
    
    
    @abstractmethod
    def get_continuation_value(self, price: float, price_info: PriceInfo) -> float:
        """
        Get value of waiting given current price and price info
        To be implemented by specific solution methods
        """
        pass
    
    def should_purchase(self, price: float, price_info: PriceInfo) -> bool:
        """
        Decide whether to purchase at current price
        Common decision rule across all forward-looking consumers:
        purchase if immediate utility exceeds continuation value
        """
        if self.has_purchased:
            return False
            
        immediate_util = self.utility(price)
        continuation_value = self.get_continuation_value(price, price_info)
        
        if immediate_util >= continuation_value:
            self.has_purchased = True
            return True
            
        return False

class InfiniteHorizonConsumer(ForwardLookingConsumerBase):
    """Implementation using asymptotic result (δ -> 0)"""
    
    def get_continuation_value(self, price: float, price_info: PriceInfo) -> float:
        # Calculate option value from Corollary 3.2
        Q = sum(prob for p, prob in zip(price_info.prices, price_info.probabilities) 
                if p < price)
        
        if Q > 0:
            E = sum(prob * (price - p) for p, prob in zip(price_info.prices, price_info.probabilities)
                    if p < price) / Q
        else:
            E = 0
            
        return (self.gamma / (1 - self.gamma)) * Q * E

class FiniteHorizonDPConsumer(ForwardLookingConsumerBase):
    """Implementation using dynamic programming"""
    
    def __init__(self, value: float, gamma: float, max_horizon: int = 100):
        super().__init__(value, gamma)
        self.max_horizon = max_horizon
        self.value_cache: Dict[Tuple[float, int], float] = {}
        
    def get_continuation_value(self, price: float, price_info: PriceInfo) -> float:
        """Compute continuation value using backward induction"""
        self.value_cache.clear()
        
        def value_to_go(p: float, t: int) -> float:
            if t == 0:
                # Terminal value - assume can buy at post-experiment price
                return self.utility(price_info.post_experiment_price) if price_info.post_experiment_price else 0
                
            key = (p, t)
            if key in self.value_cache:
                return self.value_cache[key]
                
            # Immediate purchase value
            purchase_value = self.utility(p)
            
            # Expected future value if wait
            future_value = 0
            # Experiment continues
            if t > 1:
                for next_p, prob in zip(price_info.prices, price_info.probabilities):
                    future_value += prob * value_to_go(next_p, t-1)
                future_value *= (1 - price_info.delta)
            
            # Experiment ends
            if price_info.post_experiment_price is not None:
                future_value += price_info.delta * value_to_go(price_info.post_experiment_price, 0)
                
            future_value *= self.gamma
            
            optimal_value = max(purchase_value, future_value)
            self.value_cache[key] = optimal_value
            return optimal_value
            
        return value_to_go(price, self.max_horizon)
