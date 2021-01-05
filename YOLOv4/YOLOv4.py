from PASCAL_VOC import load_data
from fastestimator.op.numpyop import NumpyOp
from fastestimator.op.numpyop.meta import Sometimes
from fastestimator.op.numpyop.multivariate import HorizontalFlip
from fastestimator.op.numpyop.univariate import Normalize, ReadImage

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, activations, backend, layers, regularizers
from tensorflow.keras.layers import Layer

class BBoxFormat(NumpyOp):
    def forward(self, data, state):
        # Converting the PascalVOC data format to MSCOCO format for compatibility with YOLOv4
        bbox = np.array(data, dtype=np.float32)
        w = bbox[:, 2] - bbox[:, 0]
        h = bbox[:, 1] - bbox[:, 3]

        bbox[:, 0] = (bbox[:, 0] + bbox[:, 2])/2
        bbox[:, 1] = (bbox[:, 1] + bbox[:, 3])/2
        bbox[:, 2] = w
        bbox[:, 3] = h
        return bbox


class Mish(Layer):
    def call(self, x):
        return x * activations.tanh(activations.softplus(x))


def _conv_block(x,
                num_filters,
                kernel_size=(9, 9),
                strides=(1, 1),
                activation="mish",
                kernel_regularizer=regularizers.l2(0.0005)):
    if isinstance(strides, int):
        strides = (strides, strides)
    if strides[0] == 2:
        x = layers.ZeroPadding2D(((1, 0), (1, 0)))(x)

    x = layers.Conv2D(filters=num_filters,
                      kernel_size=kernel_size,
                      padding="same" if strides[0] == 1 else "valid",
                      strides=strides,
                      use_bias=not activation,
                      kernel_regularizer=kernel_regularizer,
                      kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                      bias_initializer=tf.constant_initializer(0.0))(x)
    if activation is not None:
        x = layers.BatchNormalization()(x)

    if activation == "mish":
        x = Mish()(x)
    elif activation == "leaky":
        x = layers.LeakyReLU(alpha=0.1)(x)
    elif activation == "relu":
        x = layers.ReLU()(x)

    return x


def _resblock(x, num_iter, num_filters1, num_filters2, activation="mish", kernel_regularizer=None):
    for _ in range(num_iter):
        res = _conv_block(x,
                          num_filters=num_filters1,
                          kernel_size=1,
                          activation=activation,
                          kernel_regularizer=kernel_regularizer)
        res = _conv_block(res,
                          num_filters=num_filters2,
                          kernel_size=3,
                          activation=activation,
                          kernel_regularizer=kernel_regularizer)
        x = layers.Add()([x, res])
    return x


def _spp_block(x):
    pool1 = layers.MaxPooling2D((13, 13), strides=1, padding="same")(x)
    pool2 = layers.MaxPooling2D((9, 9), strides=1, padding="same")(x)
    pool3 = layers.MaxPooling2D((5, 5), strides=1, padding="same")(x)
    x = layers.Concatenate(axis=-1)([pool1, pool2, pool3, x])
    return x


def _CSPResnet(x, num_iter, num_filters1, num_filters2, activation="mish", kernel_regularizer=None):
    x = _conv_block(x,
                    num_filters=num_filters1,
                    kernel_size=3,
                    strides=2,
                    activation=activation,
                    kernel_regularizer=kernel_regularizer)

    x_1 = _conv_block(x,
                      num_filters=num_filters2,
                      kernel_size=1,
                      activation=activation,
                      kernel_regularizer=kernel_regularizer)
    x_1 = _resblock(x_1,
                    num_iter,
                    num_filters1 // 2,
                    num_filters2,
                    activation=activation,
                    kernel_regularizer=kernel_regularizer)
    x_1 = _conv_block(x_1,
                      num_filters=num_filters2,
                      kernel_size=1,
                      activation=activation,
                      kernel_regularizer=kernel_regularizer)

    x_2 = _conv_block(x,
                      num_filters=num_filters2,
                      kernel_size=1,
                      activation=activation,
                      kernel_regularizer=kernel_regularizer)

    x = layers.Concatenate(axis=-1)([x_1, x_2])

    x = _conv_block(x,
                    num_filters=num_filters1,
                    kernel_size=1,
                    activation=activation,
                    kernel_regularizer=kernel_regularizer)

    return x


def _CSPDarknet53(x, activation1="mish", activation2="leaky", kernel_regularizer=None):
    x = _conv_block(x, num_filters=32, kernel_size=3, activation=activation1, kernel_regularizer=kernel_regularizer)
    x = _CSPResnet(x,
                   num_iter=1,
                   num_filters1=64,
                   num_filters2=64,
                   activation=activation1,
                   kernel_regularizer=kernel_regularizer)
    x = _CSPResnet(x,
                   num_iter=2,
                   num_filters1=128,
                   num_filters2=64,
                   activation=activation1,
                   kernel_regularizer=kernel_regularizer)
    x = _CSPResnet(x,
                   num_iter=8,
                   num_filters1=256,
                   num_filters2=128,
                   activation=activation1,
                   kernel_regularizer=kernel_regularizer)

    path1 = x

    x = _CSPResnet(x,
                   num_iter=8,
                   num_filters1=512,
                   num_filters2=256,
                   activation=activation1,
                   kernel_regularizer=kernel_regularizer)

    path2 = x

    x = _CSPResnet(x,
                   num_iter=4,
                   num_filters1=1024,
                   num_filters2=512,
                   activation=activation1,
                   kernel_regularizer=kernel_regularizer)
    x = _conv_block(x, num_filters=512, kernel_size=1, activation=activation2, kernel_regularizer=kernel_regularizer)
    x = _conv_block(x, num_filters=1024, kernel_size=3, activation=activation2, kernel_regularizer=kernel_regularizer)
    x = _conv_block(x, num_filters=512, kernel_size=1, activation=activation2, kernel_regularizer=kernel_regularizer)
    x = _spp_block(x)
    x = _conv_block(x, num_filters=512, kernel_size=1, activation=activation2, kernel_regularizer=kernel_regularizer)
    x = _conv_block(x, num_filters=1024, kernel_size=3, activation=activation2, kernel_regularizer=kernel_regularizer)
    x = _conv_block(x, num_filters=512, kernel_size=1, activation=activation2, kernel_regularizer=kernel_regularizer)

    path3 = x
    return (path1, path2, path3)


def _PANet(x, num_classes, activation="leaky", kernel_regularizer=None):
    route1, route2, route3 = x
    x_pt1 = _conv_block(route2,
                        num_filters=256,
                        kernel_size=1,
                        activation=activation,
                        kernel_regularizer=kernel_regularizer)
    x_2 = _conv_block(route3,
                      num_filters=256,
                      kernel_size=1,
                      activation=activation,
                      kernel_regularizer=kernel_regularizer)
    x_pt2 = layers.UpSampling2D(interpolation="bilinear")(x_2)
    x_1 = layers.Concatenate(axis=-1)([x_pt1, x_pt2])

    x_1 = _conv_block(x_1, num_filters=256, kernel_size=1, activation=activation, kernel_regularizer=kernel_regularizer)
    x_1 = _conv_block(x_1, num_filters=512, kernel_size=3, activation=activation, kernel_regularizer=kernel_regularizer)
    x_1 = _conv_block(x_1, num_filters=256, kernel_size=1, activation=activation, kernel_regularizer=kernel_regularizer)
    x_1 = _conv_block(x_1, num_filters=512, kernel_size=3, activation=activation, kernel_regularizer=kernel_regularizer)
    x_1 = _conv_block(x_1, num_filters=256, kernel_size=1, activation=activation, kernel_regularizer=kernel_regularizer)

    x_2 = _conv_block(x_1, num_filters=128, kernel_size=1, activation=activation, kernel_regularizer=kernel_regularizer)
    x_pt2 = layers.UpSampling2D(interpolation="bilinear")(x_2)

    x_pt1 = _conv_block(route1,
                        num_filters=128,
                        kernel_size=1,
                        activation=activation,
                        kernel_regularizer=kernel_regularizer)

    x_2 = layers.Concatenate(axis=-1)([x_pt1, x_pt2])

    x_2 = _conv_block(x_2, num_filters=128, kernel_size=1, activation=activation, kernel_regularizer=kernel_regularizer)
    x_2 = _conv_block(x_2, num_filters=256, kernel_size=3, activation=activation, kernel_regularizer=kernel_regularizer)
    x_2 = _conv_block(x_2, num_filters=128, kernel_size=1, activation=activation, kernel_regularizer=kernel_regularizer)
    x_2 = _conv_block(x_2, num_filters=256, kernel_size=3, activation=activation, kernel_regularizer=kernel_regularizer)
    x_2 = _conv_block(x_2, num_filters=128, kernel_size=1, activation=activation, kernel_regularizer=kernel_regularizer)

    pred_s = _conv_block(x_2,
                         num_filters=256,
                         kernel_size=3,
                         activation=activation,
                         kernel_regularizer=kernel_regularizer)
    pred_s = _conv_block(pred_s,
                         num_filters=3 * (num_classes + 5),
                         kernel_size=1,
                         activation=None,
                         kernel_regularizer=kernel_regularizer)
    x_2 = _conv_block(x_2,
                      num_filters=256,
                      kernel_size=3,
                      strides=2,
                      activation=activation,
                      kernel_regularizer=kernel_regularizer)
    x_2 = layers.Concatenate(axis=-1)([x_2, x_1])

    x_2 = _conv_block(x_2, num_filters=256, kernel_size=1, activation=activation, kernel_regularizer=kernel_regularizer)
    x_2 = _conv_block(x_2, num_filters=512, kernel_size=3, activation=activation, kernel_regularizer=kernel_regularizer)
    x_2 = _conv_block(x_2, num_filters=256, kernel_size=1, activation=activation, kernel_regularizer=kernel_regularizer)
    x_2 = _conv_block(x_2, num_filters=512, kernel_size=3, activation=activation, kernel_regularizer=kernel_regularizer)
    x_2 = _conv_block(x_2, num_filters=256, kernel_size=1, activation=activation, kernel_regularizer=kernel_regularizer)
    pred_m = _conv_block(x_2,
                         num_filters=512,
                         kernel_size=3,
                         activation=activation,
                         kernel_regularizer=kernel_regularizer)
    pred_m = _conv_block(pred_m,
                         num_filters=3 * (num_classes + 5),
                         kernel_size=1,
                         activation=None,
                         kernel_regularizer=kernel_regularizer)

    x_2 = _conv_block(x_2,
                      num_filters=512,
                      kernel_size=3,
                      strides=2,
                      activation=activation,
                      kernel_regularizer=kernel_regularizer)
    x_2 = layers.Concatenate(axis=-1)([x_2, route3])

    x_2 = _conv_block(x_2, num_filters=512, kernel_size=1, activation=activation, kernel_regularizer=kernel_regularizer)
    x_2 = _conv_block(x_2,
                      num_filters=1024,
                      kernel_size=3,
                      activation=activation,
                      kernel_regularizer=kernel_regularizer)
    x_2 = _conv_block(x_2, num_filters=512, kernel_size=1, activation=activation, kernel_regularizer=kernel_regularizer)
    x_2 = _conv_block(x_2,
                      num_filters=1024,
                      kernel_size=3,
                      activation=activation,
                      kernel_regularizer=kernel_regularizer)
    x_2 = _conv_block(x_2, num_filters=512, kernel_size=1, activation=activation, kernel_regularizer=kernel_regularizer)
    pred_l = _conv_block(x_2,
                         num_filters=1024,
                         kernel_size=3,
                         activation=activation,
                         kernel_regularizer=kernel_regularizer)
    pred_l = _conv_block(pred_l,
                         num_filters=3 * (num_classes + 5),
                         kernel_size=1,
                         activation=None,
                         kernel_regularizer=kernel_regularizer)

    return pred_s, pred_m, pred_l


class YOLOv3Head(Model):
    def __init__(self, anchors, num_classes, xysclaes):
        super(YOLOv3Head, self).__init__(name="YOLOv3Head")
        self.a_half = None
        self.anchors = anchors
        self.grid_coord = []
        self.grid_size = None
        self.image_width = None
        self.num_classes = num_classes
        self.scales = xysclaes

        self.reshape0 = layers.Reshape((-1, ))
        self.reshape1 = layers.Reshape((-1, ))
        self.reshape2 = layers.Reshape((-1, ))

        self.concat0 = layers.Concatenate(axis=-1)
        self.concat1 = layers.Concatenate(axis=-1)
        self.concat2 = layers.Concatenate(axis=-1)

    def build(self, input_shape):
        # None, g_height, g_width,
        #       (xywh + conf + num_classes) * (# of anchors)

        # g_width, g_height
        _size = [(shape[2], shape[1]) for shape in input_shape]

        self.reshape0.target_shape = (
            _size[0][1],
            _size[0][0],
            3,
            5 + self.num_classes,
        )
        self.reshape1.target_shape = (
            _size[1][1],
            _size[1][0],
            3,
            5 + self.num_classes,
        )
        self.reshape2.target_shape = (
            _size[2][1],
            _size[2][0],
            3,
            5 + self.num_classes,
        )

        self.a_half = [
            tf.constant(
                0.5,
                dtype=tf.float32,
                shape=(1, _size[i][1], _size[i][0], 3, 2),
            ) for i in range(3)
        ]

        for i in range(3):
            xy_grid = tf.meshgrid(tf.range(_size[i][0]), tf.range(_size[i][1]))
            xy_grid = tf.stack(xy_grid, axis=-1)
            xy_grid = xy_grid[tf.newaxis, :, :, tf.newaxis, :]
            xy_grid = tf.tile(xy_grid, [1, 1, 1, 3, 1])
            xy_grid = tf.cast(xy_grid, tf.float32)
            self.grid_coord.append(xy_grid)

        self.grid_size = tf.convert_to_tensor(_size, dtype=tf.float32)
        self.image_width = tf.convert_to_tensor(_size[0][0] * 8.0, dtype=tf.float32)

    def call(self, x):
        raw_s, raw_m, raw_l = x

        raw_s = self.reshape0(raw_s)
        raw_m = self.reshape1(raw_m)
        raw_l = self.reshape2(raw_l)

        txty_s, twth_s, conf_s, prob_s = tf.split(raw_s, (2, 2, 1, self.num_classes), axis=-1)
        txty_m, twth_m, conf_m, prob_m = tf.split(raw_m, (2, 2, 1, self.num_classes), axis=-1)
        txty_l, twth_l, conf_l, prob_l = tf.split(raw_l, (2, 2, 1, self.num_classes), axis=-1)

        txty_s = activations.sigmoid(txty_s)
        txty_s = (txty_s - self.a_half[0]) * self.scales[0] + self.a_half[0]
        bxby_s = (txty_s + self.grid_coord[0]) / self.grid_size[0]
        txty_m = activations.sigmoid(txty_m)
        txty_m = (txty_m - self.a_half[1]) * self.scales[1] + self.a_half[1]
        bxby_m = (txty_m + self.grid_coord[1]) / self.grid_size[1]
        txty_l = activations.sigmoid(txty_l)
        txty_l = (txty_l - self.a_half[2]) * self.scales[2] + self.a_half[2]
        bxby_l = (txty_l + self.grid_coord[2]) / self.grid_size[2]

        conf_s = activations.sigmoid(conf_s)
        conf_m = activations.sigmoid(conf_m)
        conf_l = activations.sigmoid(conf_l)

        prob_s = activations.sigmoid(prob_s)
        prob_m = activations.sigmoid(prob_m)
        prob_l = activations.sigmoid(prob_l)

        bwbh_s = (self.anchors[0] / self.image_width) * backend.exp(twth_s)
        bwbh_m = (self.anchors[1] / self.image_width) * backend.exp(twth_m)
        bwbh_l = (self.anchors[2] / self.image_width) * backend.exp(twth_l)

        pred_s = self.concat0([bxby_s, bwbh_s, conf_s, prob_s])
        pred_m = self.concat1([bxby_m, bwbh_m, conf_m, prob_m])
        pred_l = self.concat2([bxby_l, bwbh_l, conf_l, prob_l])

        return pred_s, pred_m, pred_l


def YOLOv4(anchors,
           num_classes,
           xyscales,
           input_shape,
           activation1="mish",
           activation2="leaky",
           kernel_regularizer=None):
    x = layers.Input(shape=input_shape)
    x = _CSPDarknet53(x, activation1=activation1, activation2=activation2, kernel_regularizer=kernel_regularizer)
    x = _PANet(x, num_classes, activation=activation2, kernel_regularizer=kernel_regularizer)
    x = YOLOv3Head(anchors=anchors, num_classes=num_classes, xysclaes=xyscales)(x)
    return x


def get_estimator(data_dir=None,
                  model_dir=tempfile.mkdtemp(),
                  batch_size=4,
                  epochs=10,
                  max_train_steps_per_epoch=None,
                  max_eval_steps_per_epoch=None,
                  image_size=(608, 416, 3),
                  num_classes=20):
    # pipeline
    train_ds, eval_ds = load_data(root_dir=data_dir)

    pipeline = fe.Pipeline(
        train_data=train_ds,
        eval_data=eval_ds,
        batch_size=batch_size,
        ops=[
            ReadImage(inputs="image", outputs="image"),
            BBoxFormat(inputs="bbox", outputs="bbox"),
            Sometimes(
                HorizontalFlip(mode="train",
                               image_in="image",
                               image_out="image",
                               bbox_in="bbox",
                               bbox_out="bbox",
                               bbox_params='coco')),
            Normalize(inputs="image", outputs="image", mean=1.0, std=1.0, max_pixel_value=127.5)
        ])

    # network
    model = fe.build(model_fn=lambda: YOLOv4(anchors = [[[12, 16], [19, 36], [40, 28]], [[36, 75], [76, 55], [72, 146]],
                                                        [[142, 110], [192, 243], [459, 401]],],
                                             num_classes=20,
                                             xyscales = [1.2, 1.1, 1.05],
                                             input_shape=image_size, ),
                     optimizer_fn=lambda: tf.optimizers.SGD(momentum=0.9))

    return None