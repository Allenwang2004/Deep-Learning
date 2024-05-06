# Task 1: 用DNN學習重建一個帶限制函數
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random

# 生成帶限制函數的數據集
np.random.seed(42) # Set random seed for reproducibility
N = 512  # Total number of points
M = 512  # Number of training points
band_limit = 50  # Band-limit
coefficients = np.zeros(N, dtype=np.complex128) # 生成帶限制函數的傅立葉系數 用來生成帶限制函數 
coefficients[1:band_limit] = np.random.randn(band_limit-1) + 1j * np.random.randn(band_limit-1) # 生成隨機傅立葉系數 用來生成帶限制函數
coefficients[-(band_limit-1):] = coefficients[1:band_limit][::-1].conj()  # 生成對稱的傅立葉系數 用來生成帶限制函數
time_domain_function = np.fft.ifft(coefficients).real  # 生成帶限制函數

# 定義DNN模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01), input_shape=(1,)), #  L2 regularization
    tf.keras.layers.Dense(314, activation='relu'),
    tf.keras.layers.Dense(314, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'), 
    tf.keras.layers.Dense(1, activation='linear') # 最後一層的激活函數為線性函數 使得輸出值不受限制
])

def normalized(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x)) # 正規化函數值 使其在 0 到 1 之間 用來更好的訓練模型

# 編譯模型
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse') # 使用 Adam 優化器 並使用均方誤差作為損失函數   
time_domain_function = normalized(time_domain_function) # 正規化函數值 使其在 0 到 1 之間 

# 訓練模型
x_train = np.linspace(1, N, M).reshape(-1, 1)  # 生成訓練數據 用來訓練模型
x_train = x_train.reshape(-1) # 將數據展平 為了更好的訓練效果
y_train = time_domain_function[x_train.astype(int) - 1].reshape(-1) # 生成訓練標籤 用來訓練模型
xy_combined = list(zip(x_train, y_train)) # 將數據集合併
random.shuffle(xy_combined) # 打亂數據集 為了更好的訓練效果 防止模型過擬合
x_train, y_train = zip(*xy_combined) # 恢復數據集

x_train = np.array(x_train) # 將數據集轉換為 numpy 類型
y_train = np.array(y_train) # 將數據集轉換為 numpy 類型
x_train = x_train.reshape(-1, 1) # 將數據集轉換為二維數據
y_train = y_train.reshape(-1, 1) # 將數據集轉換為二維數據

history = model.fit(x_train, y_train, epochs=2000, batch_size=32, verbose=1) # 訓練模型

# 用訓練好的模型預測
x_test = np.linspace(1, N, N).reshape(-1, 1) # 生成測試數據 用來預測
y_pred = model.predict(x_test) # 預測

# 繪製訓練過程中的損失曲線
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# 繪製結果
plt.figure(figsize=(10, 6))
plt.plot(time_domain_function, label='True Function') # 繪製真實函數
plt.plot(y_pred, label='DNN Reconstruction', linestyle='--')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.title('Task 1: DNN Learns to Reconstruct a Band-limited Function')
plt.legend()
plt.show()

# 評估模型準確度
y_pred_train = model.predict(x_train)
mse = np.mean((y_pred_train - y_train) ** 2)
print(f'Mean Squared Error (MSE) on Training Data: {mse}')