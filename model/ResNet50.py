import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model


def build_resnet50(inputs):
    """
    构建 ResNet50 
    :param inputs: 输入张量
    :return: ResNet50 
    """
    def conv_block(input_tensor, filters, kernel_size=3, stride=1, conv_shortcut=False):
        """
        卷积块，包含卷积层、批归一化和激活函数。
        当 conv_shortcut 为 True 时，使用卷积操作对输入进行快捷连接，否则使用恒等连接。
        :param input_tensor: 输入张量
        :param filters: 卷积核数量
        :param kernel_size: 卷积核大小，默认为 3
        :param stride: 步长，默认为 1
        :param conv_shortcut: 是否使用卷积快捷连接，默认为 False
        :return: 处理后的张量
        """
        if conv_shortcut:
            shortcut = layers.Conv2D(4 * filters, 1, strides=stride)(input_tensor)
            shortcut = layers.BatchNormalization()(shortcut)
        else:
            shortcut = input_tensor

        x = layers.Conv2D(filters, 1, strides=stride)(input_tensor)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters, kernel_size, padding='SAME', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(4 * filters, 1)(x)
        x = layers.BatchNormalization()(x)

        x = layers.Add()([shortcut, x])
        x = layers.Activation('relu')(x)
        return x


    def identity_block(input_tensor, filters, kernel_size=3):
        """
        恒等块，输入和输出的维度相同，通过残差连接直接相加。
        :param input_tensor: 输入张量
        :param filters: 卷积核数量
        :param kernel_size: 默认为 3
        :return: 处理后的张量
        """
        shortcut = input_tensor

        x = layers.Conv2D(filters, 1)(input_tensor)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters, kernel_size, padding='SAME', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(4 * filters, 1)(x)
        x = layers.BatchNormalization()(x)

        x = layers.Add()([shortcut, x])
        x = layers.Activation('relu')(x)
        return x


    x = layers.ZeroPadding2D(padding=(3, 3))(inputs)
    x = layers.Conv2D(64, 7, strides=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(3, strides=2)(x)

    x = conv_block(x, 64, conv_shortcut=True)
    x = identity_block(x, 64)
    x = identity_block(x, 64)

    x = conv_block(x, 128, stride=2, conv_shortcut=True)
    x = identity_block(x, 128)
    x = identity_block(x, 128)
    x = identity_block(x, 128)

    x = conv_block(x, 256, stride=2, conv_shortcut=True)
    x = identity_block(x, 256)
    x = identity_block(x, 256)
    x = identity_block(x, 256)
    x = identity_block(x, 256)

    x = conv_block(x, 512, stride=2, conv_shortcut=True)
    x = identity_block(x, 512)
    x = identity_block(x, 512)

    backbone = Model(inputs, x)
    return backbone