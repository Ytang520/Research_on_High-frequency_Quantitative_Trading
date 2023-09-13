from .IFactor import IFactor
import numpy as np

class oir(IFactor):

    describe = {
        "name": "oir",
        "datas": [
            "depth5"
        ],
        "params": [
            {
                "name": "log",
                "default_value": True
            }
        ],
        "description": """
        Orderbook Imbalance Ratio.
        """
    }
    
    def main(cls, **kwargs):
        depth5 = kwargs.get("datas").get("depth5")
        if kwargs.get("params",{}).get("log"):
            return 2 * np.log(depth5['bv1'] + 1) / (np.log(depth5['av1'] + 1) + np.log(depth5['bv1'] + 1)) - 1
        else:
            return 2 * depth5['bv1'] / (depth5['av1'] + depth5['bv1']) - 1 