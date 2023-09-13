import numpy as np
import math
from sklearn.preprocessing import StandardScaler
from Factor_self import *
import tensorflow as tf


def data_import(path, profit=None, diff_n=500):
    '''
    读入数据并进行数据预处理，此处默认取第一级价差和压差下的因子（具体因子可从代码中更改 or 加入修饰函数）
    其中, 'bid1_vol', 'bid1_price' 为买方一级出价和以及买量的名称，其余级别和卖家可类比
    input:
        path: 文件路径
        diff_n: 表示选择的差分值
    output:
        path: 加入因子值的df,
    '''

    if path == 0:
        df = pd.read_csv(path, nrows=10000)
    else:
        df = pd.read_hdf(path, key="data", start=0, stop=500000)
    df['delta'] = df['bid1_vol'] - df['ask1_vol']  # 表示两边一级买卖压差
    # print(df.columns)

    # 建议提前得到 profit series
    if not profit:
        r = np.zeros(df.shape[0])
        df['profit'] = pd.Series(r)
        for i in range(len(df) - 2000):  # profit 对应每个feature的值
            df.iloc[i, -1] = (math.log((df.loc[i + 2000, 'ask1_price'] + df.loc[i + 2000, 'bid1_price']) / 2) -
                              math.log((df.loc[i, 'ask1_price'] + df.loc[i, 'bid1_price']) / 2))
    else:
        df['profit'] = profit

    # 此处默认取第一级因子
    df['voi'] = voi(ob=df, diff_n=diff_n)
    # print(lm.shape)
    #  = lm
    df['voi_2'] = voi(ob=df, ask_volume='ask2_vol', bid_volume='bid2_vol',
                      ask_price='ask2_price', bid_price='bid2_price', diff_n=diff_n)
    df['oir'] = oir(ob=df, diff_n=diff_n)
    df['oir_2'] = oir(ob=df, ask_volume='ask2_vol', bid_volume='bid2_vol',
                      ask_price='ask2_price', bid_price='bid2_price', diff_n=diff_n)

    df['bid_ask_spread'] = bid_ask_spread(ob=df, diff_n=diff_n)

    return df


def data_preprocess(data, name=None, start=4000, end=10000, window=2000, interval=10, step=30, whether_sign=False):
    '''
    数据预处理，包括正规化，数据切片;
    input:
        data: 输入数据后得到的dataframe，shape为 (length, features); 默认含有特征 profit
        name: 表示需要使用的因子名称, 默认为 None
        start: 表示取用数据的起点，至少需要 > diff_n，因为部分因子前 diff_n 长度取0，且开盘值容易动荡
        end: 表示取用数据的终点，默认为20000
        window: 时间窗, 即每次预测时使用的length长度, 默认取100
        interval: 每个时间窗间所间隔的时间步，默认为1；若值(取整数)越大，其间隔时间步越长，储存时内存消耗越少
        step: 表示需要的n步预测结果，默认取30
        whether_sign : 表示是否判断涨跌，默认值为False
    output:
        x: shape = (num_of_window, window_length, features) ,此处 features 指使用的因子
        y: shape = (num_of_window, 1)
    '''

    df = data
    if 'profit' not in df.columns:
        print('还没得到利润值，返回空。请对 data 并入利润值序列，')
        return None

    if name:
        # print(name)
        data_needed_x = df[name].values[start:end, :]
        # data_needed_x = data.drop('profit', axis=1).values
        data_needed_y = df['profit'].iloc[start:end].values

        normalized_train_x = StandardScaler().fit_transform(data_needed_x)  # 标准化

        length = int((normalized_train_x.shape[0] - window + 1) / interval)
        features = normalized_train_x.shape[1]
        x = np.zeros([length, window, features])
        y = np.zeros(length)
        if not whether_sign:
            for i in range(length):
                x[i, :, :] = normalized_train_x[i:i + window, :]
                # y[i] = data_needed_y[i]
                normalized_train_y = np.squeeze(StandardScaler().fit_transform(np.reshape(data_needed_y, (-1, 1))),
                                                axis=-1)
                y[i] = normalized_train_y[i]
        else:
            for i in range(length):
                x[i, :, :] = normalized_train_x[i:i + window, :]
                if data_needed_y[i] >= 0:
                    y[i] = 1
                else:
                    y[i] = 0
                # y[i] = data_needed_y[i + step]

        # 转化为tensor
        x = tf.convert_to_tensor(x, dtype=tf.float64)
        y = tf.convert_to_tensor(y, dtype=tf.float64)
        return x, y

    else:
        print('没有传入需要预处理的列名，即没有取到特征，返回空')
        return None