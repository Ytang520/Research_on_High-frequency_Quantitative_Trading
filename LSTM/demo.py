from lstm_part_new import *
import matplotlib.pyplot as plt

whether_sign = False
import time

t = time.time()
path = 'BINANCE_SWAP_BTC-USDT_DEPTH5_2022_02_28.hdf'  # Data_source
df = data_import(path, diff_n=2000)

x, y = data_preprocess(df, name=['voi', 'oir', 'delta', 'voi_2', 'oir_2'], end=200000, whether_sign=whether_sign)
test_x, test_y = data_preprocess(df, name=['voi', 'oir', 'delta', 'voi_2', 'oir_2'], start=200000, end=250000,
                                 whether_sign=whether_sign)
y = tf.reshape(y, (-1, 1))
test_y = tf.reshape(test_y, (-1, 1))
model = lstm(x, y, batchz=40, eps=50, units=[50, 10, 1], whether_sign=whether_sign)

# Plot: loss
lamda = model.history.history['loss']

plt.plot(model.history.history['loss'], label='train')
plt.plot(model.history.history['val_loss'], label='val')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

t_2 = time.time()
print('运行时间: ', t_2 - t)

# Metrics
if not whether_sign:
    epsilon = 1e-6
    y_pred = model.predict(test_x)
    test_loss_mse = tf.reduce_mean(tf.keras.losses.mse(y_pred, test_y))
    test_loss_mae = tf.reduce_mean(tf.keras.losses.mae(y_pred, test_y))
    test_loss_mape = tf.reduce_mean(tf.keras.losses.mape(y_pred, test_y + epsilon))
    print('test_loss__mse', test_loss_mse, 'test_loss__mae', test_loss_mae, 'test_loss_mape', test_loss_mape)

else:
    test_loss, test_acc = model.evaluate(test_x, test_y)
    print('test_loss', test_loss, 'Test accuracy:', test_acc)

# Plot: metrics
y_pred = model.predict(test_x)
x = np.arange(y_pred.shape[0])
plt.scatter(x, test_y, label='y_test')
plt.scatter(x, y_pred, label='y_pred')
plt.title('Model test_pic')
plt.ylabel('value')
plt.xlabel('time_step')
plt.legend()
plt.show()
