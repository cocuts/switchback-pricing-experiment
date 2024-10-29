from typing import List
from dataclasses import dataclass
import numpy as np

class BaseFirm:
    """Base class for firms"""
    def __init__(self, 
                 initial_price: float,
                 marginal_cost: float,
                 reorder_quantity: float,
                 initial_inventory: float = 1e250,
                 inventory_cost: float = 0):
        self.price = initial_price
        self.marginal_cost = marginal_cost
        self.inventory = initial_inventory
        self.reorder_quantity = reorder_quantity
        self.inventory_cost = inventory_cost
        self.market = None
        self.firm_id = None
        self.profits = []
                
    def update(self, realized_demand: float) -> float:
        """
        Update firm state based on realized demand
        Returns profit for the period
        """
        # Calculate revenue and costs
        revenue = self.price * realized_demand
        production_cost = self.marginal_cost * realized_demand
        inventory_cost = self.inventory_cost * self.inventory
        
        # Update inventory
        self.inventory -= realized_demand
        
        # Reorder if necessary
        if self.inventory < self.reorder_quantity:
            self.inventory += self.reorder_quantity
            production_cost += self.marginal_cost * self.reorder_quantity
            
        profit = revenue - production_cost - inventory_cost
        self.profits.append(profit)
        
        return profit


@dataclass
class SwitchbackConfig:
    """Configuration for switchback experiment"""
    reference_price: float  # p*
    discount_levels: List[float]  # ϵ values
    probabilities: List[float]  # q values 
    delta: float  # Experiment ending probability
    min_inventory_threshold: float  # Minimum inventory to offer discounts
    
    def __post_init__(self):
        if not np.isclose(sum(self.probabilities), 1.0):
            raise ValueError("Probabilities must sum to 1")
        if len(self.discount_levels) != len(self.probabilities) - 1:
            raise ValueError("Must have one less discount level than probabilities")

class ExperimentingFirm(BaseFirm):
    """Firm running a switchback price experiment with inventory management"""
    
    def __init__(self,
                 initial_price: float,
                 marginal_cost: float, 
                 initial_inventory: float,
                 reorder_quantity: float,
                 switchback_config: SwitchbackConfig):
        super().__init__(initial_price, marginal_cost, initial_inventory, reorder_quantity)
        self.config = switchback_config
        self.experiment_active = True
        self.stock_out_periods = 0
        self.last_reorder_period = 0
        
    def get_experimental_price(self) -> float:
        """Sample price for current period according to experiment design"""
        if not self.experiment_active:
            return self.price
            
        # Check if experiment ends
        if np.random.random() < self.config.delta:
            self.experiment_active = False
            return self.price
            
        # If inventory is too low, don't offer discounts
        if self.inventory < self.config.min_inventory_threshold:
            return self.config.reference_price
            
        # Sample price level
        price_idx = np.random.choice(len(self.config.probabilities), 
                                   p=self.config.probabilities)
        
        if price_idx == 0:
            return self.config.reference_price
        else:
            return self.config.reference_price - self.config.discount_levels[price_idx-1]

    def update(self, realized_demand: float) -> float:
        """
        Update firm state with realized demand
        Handles inventory constraints and stockouts
        """
        # Cap actual sales at available inventory
        actual_sales = min(realized_demand, self.inventory)
        if actual_sales < realized_demand:
            self.stock_out_periods += 1
            
        # Sample next experimental price considering inventory
        self.price = self.get_experimental_price()
        
        # Calculate revenue and costs
        revenue = self.price * actual_sales
        production_cost = self.marginal_cost * actual_sales
        inventory_cost = self.inventory_cost * self.inventory
        
        # Update inventory
        self.inventory -= actual_sales
        
        # Reorder if necessary
        if self.inventory < self.reorder_quantity:
            self.inventory += self.reorder_quantity
            production_cost += self.marginal_cost * self.reorder_quantity
            self.last_reorder_period = len(self.profits)
            
        profit = revenue - production_cost - inventory_cost
        self.profits.append(profit)
        
        return profit