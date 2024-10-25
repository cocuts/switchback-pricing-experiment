from switchback_experiment import consumers
from switchback_experiment.utility_functions import LinearUnitDemand, CRRA, QuasiLinear
import matplotlib.pyplot as plt
import numpy as np

# Create price info
price_info = consumers.PriceInfo(
    prices=[100, 90, 80],
    probabilities=[0.6, 0.2, 0.2],
    delta=0.01,
    post_experiment_price=100
)

# Create consumers with different solution methods
analytical_consumer = consumers.AnalyticConsumer(
    value=95, 
    gamma=0.8
)

gridpoint_consumer = consumers.GridpointConsumer(
    utility_function=LinearUnitDemand(value=95),
    gamma=0.8,
    grid_size=100
)

crra_consumer = consumers.GridpointConsumer(
    utility_function=CRRA(gamma=1000),
    gamma=0.8,
    grid_size=100
)

quasilinear_consumer = consumers.GridpointConsumer(
    utility_function=QuasiLinear(alpha=1.5),
    gamma=0.8,
    grid_size=100
)

# Test purchase decisions
price = 100.0
print(analytical_consumer.should_purchase(price, price_info))
print(gridpoint_consumer.should_purchase(price, price_info))
print(crra_consumer.should_purchase(price, price_info))
print(quasilinear_consumer.should_purchase(price, price_info))

# Visualize utility and decisions
prices = np.linspace(70, 110, 100)
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Utility Functions and Purchase Decisions')

# Plot for analytical/linear consumer
ax = axes[0,0]
utils = [analytical_consumer.utility_function.utility(True, p) for p in prices]
conts = [analytical_consumer.get_continuation_value(p, price_info) for p in prices]
decisions = [analytical_consumer.should_purchase(p, price_info) for p in prices]

ax.plot(prices, utils, label='Immediate Utility')
ax.plot(prices, conts, label='Continuation Value')
ax.fill_between(prices, utils, conts, where=np.array(decisions), alpha=0.3, label='Purchase Region')
ax.set_title('Analytical Consumer (Linear)')
ax.legend()
ax.grid(True)

# Plot for gridpoint/linear consumer
ax = axes[0,1]
utils = [gridpoint_consumer.utility_function.utility(True, p) for p in prices]
conts = [gridpoint_consumer.get_continuation_value(p, price_info) for p in prices]
decisions = [gridpoint_consumer.should_purchase(p, price_info) for p in prices]

ax.plot(prices, utils, label='Immediate Utility')
ax.plot(prices, conts, label='Continuation Value')
ax.fill_between(prices, utils, conts, where=np.array(decisions), alpha=0.3, label='Purchase Region')
ax.set_title('Gridpoint Consumer (Linear)')
ax.legend()
ax.grid(True)

# Plot for CRRA consumer
ax = axes[1,0]
utils = [crra_consumer.utility_function.utility(True, p) for p in prices]
conts = [crra_consumer.get_continuation_value(p, price_info) for p in prices]
decisions = [crra_consumer.should_purchase(p, price_info) for p in prices]

ax.plot(prices, utils, label='Immediate Utility')
ax.plot(prices, conts, label='Continuation Value')
ax.fill_between(prices, utils, conts, where=np.array(decisions), alpha=0.3, label='Purchase Region')
ax.set_title('CRRA Consumer')
ax.legend()
ax.grid(True)

# Plot for quasilinear consumer
ax = axes[1,1]
utils = [quasilinear_consumer.utility_function.utility(True, p) for p in prices]
conts = [quasilinear_consumer.get_continuation_value(p, price_info) for p in prices]
decisions = [quasilinear_consumer.should_purchase(p, price_info) for p in prices]

ax.plot(prices, utils, label='Immediate Utility')
ax.plot(prices, conts, label='Continuation Value')
ax.fill_between(prices, utils, conts, where=np.array(decisions), alpha=0.3, label='Purchase Region')
ax.set_title('Quasilinear Consumer')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.show()