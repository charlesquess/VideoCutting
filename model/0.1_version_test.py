import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
import datetime
from ResNet50 import build_resnet50


class FasterRCNNBuilder:
    def __init__(self, input_shape=(224, 224, 3), num_classes=10, num_anchors=9):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.model = self.build_faster_rcnn()

    def build_rpn(self, inputs):
        """
        构建区域提议网络 (RPN)
        :param inputs: 输入特征图
        :return: RPN 的分类和边界框预测结果
        """
        x = layers.Conv2D(512, (3, 3), padding='same', activation='relu', name='rpn_conv')(inputs)
        rpn_class = layers.Conv2D(self.num_anchors * 2, (1, 1), activation='softmax', name='rpn_class')(x)
        rpn_bbox = layers.Conv2D(self.num_anchors * 4, (1, 1), activation='linear', name='rpn_bbox')(x)
        return rpn_class, rpn_bbox

    def roi_pooling(self, feature_map, rois, output_size=(7, 7)):
        """
        执行 RoI 池化操作
        :param feature_map: 特征图
        :param rois: 感兴趣区域
        :param output_size: 池化后的输出尺寸
        :return: 池化后的特征图
        """
        pool = tf.image.crop_and_resize(feature_map, rois, box_indices=tf.zeros(tf.shape(rois)[0], dtype=tf.int32), crop_size=output_size)
        return pool

    def build_head(self, inputs):
        """
        构建分类和回归头
        :param inputs: 输入特征图
        :return: 分类和回归结果
        """
        x = layers.Flatten()(inputs)
        x = layers.Dense(4096, activation='relu')(x)
        cls = layers.Dense(self.num_classes, activation='softmax', name='cls')(x)
        reg = layers.Dense(4, activation='linear', name='reg')(x)
        return cls, reg

    def build_faster_rcnn(self):
        """
        构建完整的 Faster R-CNN 模型
        :return: Faster R-CNN 模型
        """
        inputs = layers.Input(shape=self.input_shape)
        # 使用自定义的 ResNet50 作为骨干网络
        backbone = build_resnet50(inputs)
        feature_map = backbone.output
        rpn_class, rpn_bbox = self.build_rpn(feature_map)
        # 假设生成一些示例的 ROIs
        rois = tf.random.uniform([32, 4])
        pooled_rois = self.roi_pooling(feature_map, rois)
        cls, reg = self.build_head(pooled_rois)
        model = Model(inputs=inputs, outputs=[rpn_class, rpn_bbox, cls, reg])
        return model


    def generate_data(self, batch_size):
        """
        生成示例数据
        :param batch_size: 批量大小
        :return: 生成的图像数据和对应的标签数据
        """
        images = np.random.rand(batch_size, *self.input_shape)
        rpn_class = np.random.rand(batch_size, self.input_shape[0] // 16, self.input_shape[1] // 16, self.num_anchors * 2)
        rpn_bbox = np.random.rand(batch_size, self.input_shape[0] // 16, self.input_shape[1] // 16, self.num_anchors * 4)
        cls = np.random.rand(batch_size, self.num_classes)
        reg = np.random.rand(batch_size, 4)
        return [images], [rpn_class, rpn_bbox, cls, reg]


    def compile_model(self, optimizer='adam', loss=['categorical_crossentropy', 'mse', 'categorical_crossentropy', 'mse']):
        """
        编译模型
        :param optimizer: 优化器
        :param loss: 损失函数列表
        """
        self.model.compile(optimizer=optimizer, loss=loss)

    def train_model(self, batch_size, epochs=10):
        """
        训练模型
        :param batch_size: 批量大小
        :param epochs: 训练轮数
        """
        x, y = self.generate_data(batch_size)
        self.model.fit(x, y, epochs=epochs)


# 示例使用
if __name__ == "__main__":
    faster_rcnn_builder = FasterRCNNBuilder()
    # print(faster_rcnn_builder.model.summary())
    model = faster_rcnn_builder.model
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True)
    # 仅使用 TensorBoard 查看模型结构，不进行训练
    with tf.summary.create_file_writer(log_dir).as_default():
        tf.summary.trace_on(graph=True)
        tf.summary.trace_export(name="model_trace", step=0)