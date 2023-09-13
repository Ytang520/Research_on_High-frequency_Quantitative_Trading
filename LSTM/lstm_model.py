import tensorflow as tf
from tensorflow.keras import optimizers
from data_preprocessing import *

def lstm(x, y, units=1, batchz=32, eps=200, whether_sign=False):
    '''
    得到n层LSTM神经网络堆叠的模型, 值得注意的是units[-2]层即不返回序列，最后一层全连接（为保证网络可加速（tensorflow写时，LSTM仅能用activation='tanh'
    input:
        x: shape = (num_of_window, window_length, features) ,此处 features 指使用的因子
        y: shape = (num_of_window, 1)
        units: scalar(只能取1) 或 list结构, 表示计划每层的 output 维度, 其中list最后一位为 1; 默认值为1
        batchz: batch_size, 默认值为32
        eps: epochs, 默认值为100
        whether_sign : 表示是否判断涨跌，默认值为False

    '''

    model = tf.keras.Sequential()

    if units == 1:
        model.add(tf.keras.layers.LSTM(units))
    else:
        for step, unit in enumerate(units):
            if unit == units[-2]:
                model.add(tf.keras.layers.LSTM(unit, return_sequences=False, activation='tanh'))
                break
            else:
                model.add(tf.keras.layers.LSTM(unit, activation='tanh', return_sequences=True))

    if not whether_sign:
        earlystop_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',  # 监测的指标
            patience=20,  # 如果连续3个epoch指标没有提高，就停止训练
            verbose=1  # 显示信息
        )
        model.add(tf.keras.layers.Dense(units[-1]))
        model.compile(optimizer=optimizers.Adam(learning_rate=0.0001, amsgrad=True),
                      loss=tf.keras.losses.MeanSquaredError())
    else:
        earlystop_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',  # 监测的指标
            patience=20,  # 如果连续3个epoch指标没有提高，就停止训练
            verbose=1  # 显示信息
        )
        model.add(tf.keras.layers.Dense(units[-1], activation='sigmoid'))
        model.compile(optimizer=optimizers.Adam(learning_rate=0.0001, amsgrad=True),
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])

    model.fit(x, y, batch_size=batchz, epochs=eps, shuffle=True, validation_split=0.2, callbacks=[earlystop_callback])

    return model
