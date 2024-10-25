from typing import List, Optional
import numpy as np

class BaseFirm:
    """Base class for firms"""
    def __init__(self, 
                 initial_price: float,
                 marginal_cost: float,
                 initial_inventory: float,
                 reorder_quantity: float):
        self.price = initial_price
        self.marginal_cost = marginal_cost
        self.inventory = initial_inventory
        self.reorder_quantity = reorder_quantity
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
        inventory_cost = self.market.config.inventory_cost * self.inventory
        
        # Update inventory
        self.inventory -= realized_demand
        
        # Reorder if necessary
        if self.inventory < self.reorder_quantity:
            self.inventory += self.reorder_quantity
            production_cost += self.marginal_cost * self.reorder_quantity
            
        profit = revenue - production_cost - inventory_cost
        self.profits.append(profit)
        
        return profit

class ExperimentingFirm(BaseFirm):
    """Firm running a switchback experiment"""