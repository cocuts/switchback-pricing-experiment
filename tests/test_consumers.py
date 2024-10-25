from switchback_experiment import consumers
# Create price info
price_info = consumers.PriceInfo(
    prices=[100, 90, 80],
    probabilities=[0.6, 0.2, 0.2],
    delta=0.01,
    post_experiment_price=100
)

# Create different types of consumers
infinite_consumer = consumers.InfiniteHorizonConsumer(value=95, gamma=0.8)
dp_consumer = consumers.FiniteHorizonDPConsumer(value=95, gamma=0.8, max_horizon=100)

# Check purchase decisions
price = 100.0
print(infinite_consumer.should_purchase(price, price_info))
print(dp_consumer.should_purchase(price, price_info))
