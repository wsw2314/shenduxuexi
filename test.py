import os

import librosa
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

# 建立类别标签，不同类别对应不同的数字。
label = ['aloe', 'burger', 'cabbage', 'candied_fruits',
         'carrots', 'chips', 'chocolate', 'drinks', 'fries',
         'grapes', 'gummies', 'ice-cream', 'jelly', 'noodles', 'pickles',
         'pizza', 'ribs', 'salmon', 'soup', 'wings']
label_dict = dict(zip(label, range(len(label))))


def extract_features(path, rates=(1.0,)):
    """从音频文件中提取特征"""
    y_0, sr = librosa.load(path)
    # 缩放后的y
    y_list = [librosa.effects.time_stretch(y_0, rate=rate) for rate in rates]
    features = []
    for y in y_list:
        # 这里使用mfcc
        mel = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128).T
        features.append(np.mean(mel, axis=0))
    return np.array(features)


def extract_features_train(parent_dir, max_file=10):
    """提取训练数据集中的音频特征"""
    X, Y = [], []
    for sub_dir in label:
        _, _, filenames = next(os.walk(os.path.join(parent_dir, sub_dir)))
        for filename in tqdm(filenames[:max_file]):
            # 这里做了数据增强，拉伸系数0.5, 0.7, 1.0, 1.4, 2.0
            features = extract_features(os.path.join(parent_dir, sub_dir, filename), (0.5, 0.7, 1.0, 1.4, 2.0))
            for feature in features:
                X.append(feature)
                Y.append(label_dict[sub_dir])
    return [np.array(X), np.array(Y)]


def extract_features_test(parent_dir):
    """提取测试数据集中的音频特征"""
    X = []
    _, _, filenames = next(os.walk(parent_dir))
    for filename in tqdm(filenames):
        # 测试集不需要数据增强，所以没有传rates
        X.append(extract_features(os.path.join(parent_dir, filename))[0])
    return np.array(X)


def save_name():
    """保存测试集中文件的名称"""
    _, _, filenames = next(os.walk('C:/Users/11930/Downloads/test_a'))
    with open('path', 'w') as f:
        f.writelines([filename + '\n' for filename in filenames])


def save_features():
    """保存训练和测试特征数据"""
    save_name()
    X, Y = extract_features_train('C:/Users/11930/Downloads/train', 1000)
    print(X.shape)
    print(Y.shape)
    np.save('X.npy', X)
    np.save('Y.npy', Y)
    X_ = extract_features_test('C:/Users/11930/Downloads/test_a')
    print(X_.shape)
    np.save('X_.npy', X_)


def train_test_split(X, y, test_ratio=0.2, seed=None):
    """将数据 X 和 y 按照test_ratio分割成X_train, X_test, y_train, y_test"""
    assert X.shape[0] == y.shape[0], \
        "the size of X must be equal to the size of y"
    assert 0.0 <= test_ratio <= 1.0, \
        "test_ration must be valid"

    if seed:
        np.random.seed(seed)

    shuffled_indexes = np.random.permutation(len(X))

    test_size = int(len(X) * test_ratio)
    test_indexes = shuffled_indexes[:test_size]
    train_indexes = shuffled_indexes[test_size:]

    X_train = X[train_indexes]
    y_train = y[train_indexes]

    X_test = X[test_indexes]
    y_test = y[test_indexes]

    return X_train, X_test, y_train, y_test


def load_features():
    """加载已保存的特征数据"""
    X = np.load('X.npy')
    Y = np.load('Y.npy')
    X_ = np.load('X_.npy')
    return X, Y, X_


def classifier():
    """创建音频分类模型"""
    model = Sequential()
    # 由于n_mfcc=128，所以这里的Dense输入也为128维，激活函数使用的relu，经过尝试，效果好于tanh
    model.add(Dense(1024, input_dim=128, activation="relu"))
    # Dropout主要为了防止过拟合，这里随机去掉一半的特征进行预测
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation="relu"))
    # 音频的分类为20类
    model.add(Dense(20, activation="softmax"))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


def to_csv(model, X_, save_path='submit.csv'):
    """将模型预测结果保存为CSV文件"""
    predictions = model.predict(X_)
    preds = np.argmax(predictions, axis=1)
    preds = [label[x] for x in preds]
    path = []
    # 此处的path文件是save_name方法保存的
    with open("path") as f:
        for line in f:
            path.append(line.strip())
    result = pd.DataFrame({'name': path, 'label': preds})
    result.to_csv(save_path, index=False)

save_features()
# 从文件中加载特征
X, Y, X_ = load_features()
# 对特征进行标准化处理
train_mean = np.mean(X, axis=0)
train_std = np.std(X, axis=0)
X = (X - train_mean) / train_std
X_ = (X_ - train_mean) / train_std
# 将类别转换为one-hot
Y = to_categorical(Y)
# 训练测试数据分离
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_ratio=0.1, seed=666)
# 建立模型
model = classifier()
model.fit(X, Y, epochs=1000, batch_size=5000, validation_data=(X_test, y_test))
to_csv(model, X_, 'submit.csv')
