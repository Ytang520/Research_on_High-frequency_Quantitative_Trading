import pandas as pd


# VOI
def voi(ob, ask_volume='ask1_vol', bid_volume='bid1_vol',
        ask_price='ask1_price', bid_price='bid1_price', diff_n=300):
    """
    此表示voi因子，表示买卖方交易量差额，反应市场总体情绪和走向
    :param
        ob: orderbook,时间自第1行到第n行递增
        ask_volume, ask_price: 买方单数, 买方价格，默认买一
        bid_volume, bid_price: 卖方单数，卖方价格，默认卖一
        diff_n: 表示选择的差分数，默认为1
    :return:
        voi series: ，买卖方交易差额 (注意前 diff_n 行值为0，由原始定义决定)
    """

    ask_volume = ob.loc[:, ask_volume]
    bid_volume = ob.loc[:, bid_volume]
    ask_price = ob.loc[:, ask_price]
    bid_price = ob.loc[:, bid_price]

    b = ob.shape[0]
    voi = pd.Series(0, index=range(b))

    for i in range(b - diff_n):
        v_b = 0
        v_a = 0
        # i = i_prep + diff_n
        if bid_price[i] == bid_price[i + diff_n]:
            v_b = bid_volume.iloc[i + diff_n] - bid_volume.iloc[i]
        elif bid_price[i] < bid_price[i + diff_n]:
            v_b = bid_volume.iloc[i + diff_n]
        else:
            pass

        if ask_price[i] == ask_price[i + diff_n]:
            v_a = ask_volume.iloc[i + diff_n] - ask_volume.iloc[i]
        elif ask_price[i] < ask_price[i + diff_n]:
            v_a = ask_volume.iloc[i + diff_n]
        else:
            pass
        voi.iloc[i + diff_n] = v_b - v_a
        pass
    # print(voi.shape)
    return voi


# OIR
def oir(ob, ask_volume='ask1_vol', bid_volume='bid1_vol',
        ask_price='ask1_price', bid_price='bid1_price', diff_n=300):
    """
    此表示oir因子，表示不平衡情况，反应交易者行为特征
    :param
        ob: orderbook
        ask_volume, ask_price: 买方单数, 买方价格，默认买一
        bid_volume, bid_price: 卖方单数，卖方价格，默认卖一
        diff_n: 表示选择的差分数，默认为1
    :return:
        oir series: ，买卖方交易量不平衡情况
    """

    ask_volume = ob.loc[:, ask_volume]
    bid_volume = ob.loc[:, bid_volume]

    oir = (bid_volume - ask_volume) / (bid_volume + ask_volume)

    return oir


# Bid-Ask Spread
def bid_ask_spread(ob, ask_volume='ask1_vol', bid_volume='bid1_vol',
                   ask_price='ask1_price', bid_price='bid1_price', diff_n=300):
    """
    此表示调节因子，衡量流动性，并作调节因子
    :param
        ob: orderbook
        ask_volume, ask_price: 买方单数, 买方价格，默认买一
        bid_volume, bid_price: 卖方单数，卖方价格，默认卖一
        diff_n: 表示选择的差分数，默认为1
    :return:
        s_t: ，买卖方交易量不平衡情况
    """

    ask_price = ob.loc[:, ask_price]
    bid_price = ob.loc[:, bid_price]

    s_t = ask_price - bid_price
    return s_t

    pass
