from typing import List


class BaseConsumer:
    def __init__(self, value: float, gamma: float):
        self.value = value
        self.gamma = gamma  # Patience level
        self.has_purchased = False
    
    def utils(purchase_decision):
        return purchase_decision
    
class ForwardLookingConsumer(BaseConsumer):
    """A forward-looking buyer with a value and patience level"""