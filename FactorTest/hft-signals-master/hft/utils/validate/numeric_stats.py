import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import pacf

def numeric_stats(kwargs=None):


    if not kwargs:
        print("""input: {
            "datas": {
                    "orderbook": DataFrame,
                    "trade": DataFrame(optional),
                },
            "feature": Series,
            "ask_label": Series("optional"),
            "bid_label": Series("optional)
            }
            """)
        return

    feature = kwargs.get("feature")
    orderbook = kwargs.get("datas", {}).get("orderbook")
    trade = kwargs.get('datas', {}).get("trade")
    ask_label = kwargs.get("ask_label")
    bid_label = kwargs.get("bid_label")

    if feature is None:
        raise AttributeError("(feature) is not found in inputs")
    if orderbook is None:
        raise AttributeError("(orderbook) is not found in inputs")
        
    # result dict
    res = dict()

    # preprocessing
    mp = (orderbook['ap1'] + orderbook['bp1'])/2

    # main part
        
    ## label corrs
    if ask_label and bid_label:
        res['ask_label_corr'] = np.corrcoef(feature, ask_label)[0][1]
        res['bid_label_corr'] = np.corrcoef(feature, bid_label)[0][1]
        res['label_corrs_sym'] = _corr_sym(res['ask_label_corr'], res['bid_label_corr'])

    ## underlying corr
    res['underlying_corr'] = np.corrcoef(feature, mp)[0][1]

    ## lag 1 pacf
    _, res['lag_1_pacf'] = pacf(feature, nlags=1)

    ## hit ratio
    ap_max = orderbook['ap1'][::-1].rolling(50).max() == orderbook['ap1'][::-1]
    bp_min = orderbook['bp1'][::-1].rolling(50).min() == orderbook['bp1'][::-1]

    hit_ratio_spearman_corr = pd.concat([feature, ap_max, bp_min], axis=1).corr(method="spearman").values
    res['hit_ratio'] = abs(hit_ratio_spearman_corr[0][1]) + abs(hit_ratio_spearman_corr[0][2])

    ## std
    res['std_var'] = feature.std()

    ## skewness
    res['skewness'] = feature.skew()

    ## kurtosis
    res['kurtosis'] = feature.kurt()

    ## range_in_sigmas
    threshold_ratio = 0.005
    length_feature = len(feature)
    small_idx = int(length_feature * threshold_ratio)
    large_idx = int(length_feature * (1 - threshold_ratio))
    sorted_feature = feature.sort_values()
    small_value = sorted_feature.iloc[small_idx]
    large_value = sorted_feature.iloc[large_idx]
    range = large_value - small_value
    std = np.std(feature)
    res['range_in_sigmas'] = range / std

    return res


def _corr_sym(Ra, Rb):
    return 2 * np.tanh((abs(Ra-Rb))/(abs(Ra+Rb))) - 1

