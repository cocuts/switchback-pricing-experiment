import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Dict, List
from switchback_experiment import firms, consumers, utility_functions

def simulate_firm_consumer_interaction(
    firm: firms.ExperimentingFirm,
    list_of_consumers: List[consumers.ConsumerBase],
    n_periods: int = 100,
    growth_window: int = 3,  # Window for calculating growth rate
    base_growth_rate: float = 0.2,  # Base rate of consumer growth
    max_growth_rate: float = 0.2,  # Cap on growth rate
    min_growth_rate: float = -0.1,  # Floor on growth rate 
    seed: Optional[int] = None
) -> Dict:
    """
    Simulate interaction between firm and consumers over multiple periods
    New consumers enter market when old ones exit after purchase
    """
    if seed is not None:
        np.random.seed(seed)
        
    # Track metrics
    prices = []
    demands = []
    inventory = []
    profits = []
    population = []
    growth_rates = []
    
    # Create PriceInfo for consumers
    price_info = consumers.PriceInfo(
        prices=[firm.config.reference_price] + [
            firm.config.reference_price - d for d in firm.config.discount_levels
        ],
        probabilities=firm.config.probabilities,
        delta=firm.config.delta,
        post_experiment_price=firm.config.reference_price
    )
    
    # Initialize active consumers
    active_consumers = list_of_consumers.copy()
    recent_demands = [0] * growth_window  # Initialize demand history
    
    for t in range(n_periods):
        # Get firm's price for this period
        price = firm.get_experimental_price()
        prices.append(price)
        
        # Track consumers who purchase
        purchasers = [
            consumer for consumer in active_consumers 
            if consumer.should_purchase(price, price_info)
        ]
        period_demand = len(purchasers)
        demands.append(period_demand)
        
        # Update recent demands history
        recent_demands.pop(0)
        recent_demands.append(period_demand)
        
        # Calculate growth rate based on recent demand trend
        avg_recent_demand = np.mean(recent_demands)
        current_population = len(active_consumers)
        
        # Growth rate increases if recent demand is high relative to population
        demand_ratio = avg_recent_demand / max(current_population, 1)
        growth_rate = base_growth_rate + (demand_ratio - 0.1) * 0.5  # 0.1 is target demand ratio
        growth_rate = min(max(growth_rate, min_growth_rate), max_growth_rate)
        growth_rates.append(growth_rate)
        
        # Remove purchasers from active consumers
        active_consumers = [c for c in active_consumers if c not in purchasers]
        
        # Generate new consumers based on growth rate
        n_new = int(current_population * growth_rate)
        if n_new > 0:
            new_consumers = []
            # Add new analytical consumers
            new_consumers.extend([
                consumers.AnalyticConsumer(
                    value=np.random.uniform(80,120), 
                    gamma=0.98
                )
                for _ in range(n_new//3)
            ])
            
            # Add new quasilinear consumers
            new_consumers.extend([
                consumers.GridpointConsumer(
                    utility_function=utility_functions.QuasiLinear(alpha=2),
                    gamma=0.98,
                    grid_size=100
                )
                for _ in range(n_new//3)
            ])
            
            # Add new linear unit demand consumers
            new_consumers.extend([
                consumers.GridpointConsumer(
                    utility_function=utility_functions.LinearUnitDemand(
                        np.random.uniform(50,120)
                    ),
                    gamma=0.98,
                    grid_size=100
                )
                for _ in range(n_new - len(new_consumers))
            ])
            
            active_consumers.extend(new_consumers)
        
        population.append(len(active_consumers))
        
        # Update firm
        profit = firm.update(period_demand)
        profits.append(profit)
        inventory.append(firm.inventory)
        
    return {
        'prices': prices,
        'demands': demands,
        'inventory': inventory,
        'profits': profits,
        'population': population,
        'growth_rates': growth_rates
    }

def plot_firm_simulation(results: Dict, firm: firms.ExperimentingFirm):
    """Create visualization of firm simulation results"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle('Firm Experiment Results')
    
    # Price trajectory
    ax = axes[0,0]
    ax.plot(results['prices'], label='Price')
    ax.axhline(y=firm.config.reference_price, color='r', linestyle='--', 
               label='Reference Price')
    for d in firm.config.discount_levels:
        ax.axhline(y=firm.config.reference_price - d, color='g', linestyle=':',
                  alpha=0.5, label=f'Discount {d}')
    ax.set_title('Price Trajectory')
    ax.set_xlabel('Period')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True)
    
    # Demand realization
    ax = axes[0,1]
    ax.plot(results['demands'], label='Realized Demand')
    ax.set_title('Demand by Period')
    ax.set_xlabel('Period')
    ax.set_ylabel('Units Demanded')
    ax.legend()
    ax.grid(True)
    
    # Population dynamics
    ax = axes[0,2]
    ax.plot(results['population'], label='Active Consumers')
    ax.plot(results['growth_rates'], label='Growth Rate', alpha=0.5)
    ax.set_title('Market Size Dynamics')
    ax.set_xlabel('Period')
    ax.set_ylabel('Number of Consumers')
    ax.legend()
    ax.grid(True)
    
    # Inventory level
    ax = axes[1,0]
    ax.plot(results['inventory'], label='Inventory')
    ax.axhline(y=firm.config.min_inventory_threshold, color='r', linestyle='--',
               label='Min Threshold')
    ax.set_title('Inventory Level')
    ax.set_xlabel('Period')
    ax.set_ylabel('Units')
    ax.legend()
    ax.grid(True)
    
    # Profit
    ax = axes[1,1]
    ax.plot(results['profits'], label='Period Profit')
    ax.plot(np.cumsum(results['profits'])/np.arange(1, len(results['profits'])+1),
            label='Average Profit', linestyle='--')
    ax.set_title('Profit')
    ax.set_xlabel('Period')
    ax.set_ylabel('Profit')
    ax.legend()
    ax.grid(True)
    
    # Growth rate distribution
    ax = axes[1,2]
    ax.hist(results['growth_rates'], bins=30, alpha=0.6)
    ax.set_title('Growth Rate Distribution')
    ax.set_xlabel('Growth Rate')
    ax.set_ylabel('Frequency')
    ax.grid(True)
    
    plt.tight_layout()
    return fig, axes

if __name__ == "__main__":
    # Create firm
    config = firms.SwitchbackConfig(
        reference_price=100,
        discount_levels=[20, 40],
        probabilities=[0.9, 0.05, 0.05],
        delta=0.0,
        min_inventory_threshold=50
    )
    
    firm = firms.ExperimentingFirm(
        initial_price=100,
        marginal_cost=0,
        switchback_config=config,
        initial_inventory=1e250,
        reorder_quantity=0
    )
    
    # Create population of different consumer types
    n_consumers = 10000
    list_of_consumers = []
    
    # Add linear consumers
    list_of_consumers.extend([
        consumers.AnalyticConsumer(value=np.random.uniform(80,120), gamma=0.98)
        for _ in range(n_consumers//3)
    ])
        
    # Add quasilinear consumers
    list_of_consumers.extend([
        consumers.GridpointConsumer(
            utility_function=utility_functions.QuasiLinear(alpha=2),
            gamma=0.98,
            grid_size=100
        )
        for _ in range(n_consumers//3)
    ])

    # Add linear unit demand consumers
    list_of_consumers.extend([
        consumers.GridpointConsumer(
            utility_function=utility_functions.LinearUnitDemand(np.random.uniform(80,120)),
            gamma=0.98,
            grid_size=100
        )
        for _ in range(n_consumers//3)
    ])


    # Run simulation
    results = simulate_firm_consumer_interaction(firm, list_of_consumers, n_periods=200)
    
    # Plot results
    plot_firm_simulation(results, firm)
    plt.show()