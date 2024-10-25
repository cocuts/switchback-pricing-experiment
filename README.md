# Strategic Consumer Price Experimentation with Inventory Constraints

This project extends Wu et al.'s "Switchback Price Experiments with Forward-Looking Demand" to include inventory constraints. It provides a simulation framework for studying how strategic consumer behavior and inventory management affect firms' ability to learn demand through price experimentation.

## Overview

The simulation models a market with:
- Forward-looking consumers with heterogeneous valuations and patience levels
- Firms running switchback price experiments while managing inventory
- Market clearing mechanisms that account for strategic waiting behavior

Key components:

- `BaseConsumer` - Abstract consumer class
- `ForwardLookingConsumer` - Strategic consumer who optimizes purchase timing
- `BaseFirm` - Abstract firm class with inventory management
- `ExperimentingFirm` - Firm running switchback price experiments
- `SwitchbackExperiment` - Analyzes experimental results to estimate demand
- `Market` - Coordinates firm-consumer interactions and clears the market


## Key Features

1. **Strategic Consumer Behavior**
   - Forward-looking consumers who optimize purchase timing
   - Heterogeneous valuations and patience levels
   - Option value calculations for waiting

2. **Inventory Management**
   - Reorder points and quantities
   - Stock-out tracking
   - Inventory holding costs

3. **Price Experimentation**
   - Three-price switchback design
   - Unbiased gradient estimation
   - Same-day vs total sales analysis

4. **Market Mechanism**
   - Multi-period clearing
   - Consumer surplus calculation
   - Producer surplus calculation

## Extensions

The framework can be extended to study:
- Multi-firm competition
- Different inventory policies
- Alternative experimental designs
- Dynamic pricing strategies

## References

Wu, Y., Johari, R., Syrgkanis, V., & Weintraub, G. Y. (2024). Switchback Price Experiments with Forward-Looking Demand. arXiv preprint arXiv:2410.14904.
