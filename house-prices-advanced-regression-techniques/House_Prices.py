#
# created by OFShare at 2019-07-23
#

import tensorflow as tf
import numpy as np
import pandas as pd

slim = tf.contrib.slim

train_data = pd.read_csv('./train.csv')
test_data = pd.read_csv('./test.csv')

def parse_data():
    print('run...')
    print('train_data shape: ', train_data.shape)
    print('test_data shape: ', test_data.shape)
    print('train_data:\n', train_data.iloc[0:4,[0,1,2,3,-3,-2,-1]])
    all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
    numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
    all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
    # 标准化后，每个特征的均值变为0，所以可以直接用0来替换缺失值
    all_features[numeric_features] = all_features[numeric_features].fillna(0)
    # dummy_na=True将缺失值也当作合法的特征值并为其创建指示特征
    all_features = pd.get_dummies(all_features, dummy_na=True)

    n_train = train_data.shape[0]
    train_features = np.array(all_features[:n_train].values)
    test_features = np.array(all_features[n_train:].values)
    train_labels = np.array(train_data.SalePrice.values).reshape((-1, 1))

    return train_features, test_features, train_labels


def build_net(input_, is_training = True):
    out = tf.layers.dense(inputs = input_, units = 100)
    out = tf.layers.dropout(inputs = out, rate = 0.5, training = is_training)
    # out = tf.layers.dense(inputs = out, units = 100)
    # out = tf.layers.dense(inputs = out, units = 10)
    out = tf.layers.dense(inputs = out, units = 1)
    return out


def build_model():
    """Builds graph for model.
    Inputs:
        input_: tf.placeholder(dtype = tf.float32, shape = (H,W))
    Returns:
        g: Graph
        train_tensor: Train op for execution during training.
    """
    g = tf.Graph()
    with g.as_default():
        is_training = True
        train_features, test_features, train_labels = parse_data()
        if is_training:
            input_ = tf.convert_to_tensor(train_features, dtype=tf.float32)
        else:
            input_ = tf.convert_to_tensor(test_features, dtype=tf.float32)
        logits = build_net(input_, is_training)
        RMSE = tf.sqrt(tf.losses.mean_squared_error(logits, train_labels))
        # total_loss = tf.losses.get_total_loss(name='total_loss')
        # global_step = tf.Variable(0, trainable=False)
        global_step = tf.train.get_or_create_global_step()
        starter_learning_rate = 0.1
        learning_rate = tf.train.exponential_decay(
            starter_learning_rate,
            global_step,
            decay_steps = 1000 ,
            decay_rate = 0.96,
            staircase=True)
        opt = tf.train.GradientDescentOptimizer(learning_rate)

        # create_train_op ensures that each time we ask for the loss, the update_ops
        # are run and the gradients being computed are applied too.
        train_tensor = slim.learning.create_train_op(
            total_loss = RMSE,
            optimizer = opt)

    return g, train_tensor


def train_model():
    """Trains simple model."""
    g, train_tensor = build_model()
    with g.as_default():
        # Actually runs training.
        slim.learning.train(train_tensor, train_log_dir)


def main(unused_arg):
    train_model()


if __name__ == '__main__':
    train_log_dir = 'train_dir'
    if not tf.gfile.Exists(train_log_dir):
        tf.gfile.MakeDirs(train_log_dir)
    tf.app.run(main)
