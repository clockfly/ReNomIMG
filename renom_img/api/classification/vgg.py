from __future__ import print_function, division
import os
import numpy as np
import renom as rm
from tqdm import tqdm
from renom_img.api import Base, adddoc
from renom_img.api.classification import Classification
from renom_img.api.utility.misc.download import download
from renom_img.api.utility.load import prepare_detection_data, load_img
from renom_img.api.utility.distributor.distributor import ImageDistributor

DIR = os.path.split(os.path.abspath(__file__))[0]


def layer_factory(channel=32, conv_layer_num=2):
    layers = []
    for _ in range(conv_layer_num):
        layers.append(rm.Conv2d(channel=channel, padding=1, filter=3))
        layers.append(rm.Relu())
    layers.append(rm.MaxPool2d(filter=2, stride=2))
    return rm.Sequential(layers)


@adddoc
class VGGBase(Classification):

    def get_optimizer(self, current_epoch=None, total_epoch=None, current_batch=None, total_batch=None, **kwargs):
        if any([num is None for num in [current_epoch, total_epoch, current_batch, total_batch]]):
            return self._opt
        else:
            avg_valid_loss_list = kwargs['avg_valid_loss_list']
            if len(avg_valid_loss_list) >= 2 and avg_valid_loss_list[-1] > avg_valid_loss_list[-2] and current_batch == 0:
                self._opt._lr = self._opt._lr / 10.
            elif current_epoch == 0:
                self._opt._lr = 0.00001 + (0.001 - 0.00001) * current_batch / total_batch
            return self._opt

    def preprocess(self, x):
        """
        Returns:
            (ndarray): Preprocessed data.

        Preprocessing for VGG is follows.

        .. math::

            x_{red} -= 123.68 \\\\
            x_{green} -= 116.779 \\\\
            x_{blue} -= 103.939

        """
        x[:, 0, :, :] -= 123.68  # R
        x[:, 1, :, :] -= 116.779  # G
        x[:, 2, :, :] -= 103.939  # B
        return x

    def _freeze(self):
        self._model.block1.set_auto_update(self._train_whole_network)
        self._model.block2.set_auto_update(self._train_whole_network)
        self._model.block3.set_auto_update(self._train_whole_network)
        self._model.block4.set_auto_update(self._train_whole_network)
        self._model.block5.set_auto_update(self._train_whole_network)

    def forward(self, x):
        """
        Returns:
            (Node): Returns raw output of ${class}.

        Example:
            >>> import numpy as np
            >>> from renom_img.api.classification.vgg import ${class}
            >>>
            >>> x = np.random.rand(1, 3, 224, 224)
            >>> class_map = ["dog", "cat"]
            >>> model = ${class}(class_map)
            >>> y = model.forward(x) # Forward propagation.
            >>> y = model(x)  # Same as above result.
        """
        self._freeze()
        return self._model(x)


@adddoc
class VGG11(VGGBase):
    """VGG11 model.

    Args:
        class_map(array): Array of class names
        imsize(int or tuple): Input image size
        train_whole_network(bool): True if the overall model is trained, otherwise False
        load_pretrained_weight (bool, str): If true, pretrained weight will be
          downloaded to current directory. If string is given, pretrained weight
          will be saved as given name.

    Note:
        if the argument num_class is not 1000, last dense layer will be reset because
        the pretrained weight is trained on 1000 classification dataset.

    References:
        | Karen Simonyan, Andrew Zisserman
        | Very Deep Convolutional Networks for Large-Scale Image Recognition
        | https://arxiv.org/abs/1409.1556
        |

    """

    SERIALIZED = ("imsize", "class_map", "num_class")
    WEIGHT_URL = "http://docs.renom.jp/downloads/weights/VGG11.h5"

    def __init__(self, class_map=[], imsize=(224, 224), load_pretrained_weight=False, train_whole_network=False):
        if not hasattr(imsize, "__getitem__"):
            imsize = (imsize, imsize)

        self.imsize = imsize
        self.num_class = len(class_map)
        self.class_map = [c.encode("ascii", "ignore") for c in class_map]
        self._model = CNN_VGG11(self.num_class)
        self._train_whole_network = train_whole_network
        self._opt = rm.Sgd(0.01, 0.9)
        self.decay_rate = 0.0005

        if load_pretrained_weight:
            if isinstance(load_pretrained_weight, bool):
                load_pretrained_weight = self.__class__.__name__ + '.h5'

            if not os.path.exists(load_pretrained_weight):
                download(self.WEIGHT_URL, load_pretrained_weight)

            self._model.load(load_pretrained_weight)
            self._model.fc1.params = {}
            self._model.fc2.params = {}
            self._model.fc3.params = {}

        assert not load_pretrained_weight, "Currently pretrained weight of %s is not prepared. Please set False to `load_pretrained_weight` flag." % self.__class__.___name__


@adddoc
class VGG16(VGGBase):
    """VGG16 model.
    If the argument load_weight is True, pretrained weight will be downloaded.
    The pretrained weight is trained using ILSVRC2012.

    Args:
        class_map(array): Array of class names
        imsize(int or tuple): Input image size
        train_whole_network(bool): True if the overall model is trained, otherwise False
        load_pretrained_weight (bool, str): If true, pretrained weight will be
          downloaded to current directory. If string is given, pretrained weight
          will be saved as given name.


    Note:
        if the argument num_class is not 1000, last dense layer will be reset because
        the pretrained weight is trained on 1000 classification dataset.

    References:
        | Karen Simonyan, Andrew Zisserman
        | Very Deep Convolutional Networks for Large-Scale Image Recognition
        | https://arxiv.org/abs/1409.1556
        |

    """

    SERIALIZED = ("imsize", "class_map", "num_class")
    WEIGHT_URL = "https://renom.jp/docs/downloads/weights/Vgg16.h5"

    def __init__(self, class_map=[], imsize=(224, 224), load_pretrained_weight=False, train_whole_network=False):
        if not hasattr(imsize, "__getitem__"):
            imsize = (imsize, imsize)

        self.imsize = imsize
        self.num_class = len(class_map)
        self.class_map = [c.encode("ascii", "ignore") for c in class_map]
        self._model = CNN_VGG16(self.num_class)
        self._train_whole_network = train_whole_network
        self._opt = rm.Sgd(0.001, 0.9)
        self.decay_rate = 0.0005

        if load_pretrained_weight:
            if isinstance(load_pretrained_weight, bool):
                load_pretrained_weight = self.__class__.__name__ + '.h5'
            if not os.path.exists(load_pretrained_weight):
                download(self.WEIGHT_URL, load_pretrained_weight)

            self._model.load(load_pretrained_weight)
            self._model.fc1.params = {}
            self._model.fc2.params = {}
            self._model.fc3.params = {}


@adddoc
class VGG19(VGGBase):
    """VGG19 model.

    If the argument load_weight is True, pretrained weight will be downloaded.
    The pretrained weight is trained using ILSVRC2012.

    Args:
        class_map(array): Array of class names
        imsize(int or tuple): Input image size
        train_whole_network(bool): True if the overall model is trained, otherwise False
        load_pretrained_weight (bool, str): If true, pretrained weight will be
          downloaded to current directory. If string is given, pretrained weight
          will be saved as given name.

    Note:
        if the argument num_class is not 1000, last dense layer will be reset because
        the pretrained weight is trained on 1000 classification dataset.

    References:
        | Karen Simonyan, Andrew Zisserman
        | Very Deep Convolutional Networks for Large-Scale Image Recognition
        | https://arxiv.org/abs/1409.1556
        |

    """

    SERIALIZED = ("imsize", "class_map", "num_class")
    WEIGHT_URL = "https://renom.jp/docs/downloads/weights/Vgg19.h5"

    def __init__(self, class_map=[], imsize=(224, 224), load_pretrained_weight=False, train_whole_network=False):
        if not hasattr(imsize, "__getitem__"):
            imsize = (imsize, imsize)

        self.imsize = imsize
        self.num_class = len(class_map)
        self.class_map = [c.encode("ascii", "ignore") for c in class_map]
        self._model = CNN_VGG19(self.num_class)
        self._train_whole_network = train_whole_network
        self._opt = rm.Sgd(0.01, 0.9)
        self.decay_rate = 0.0005

        if load_pretrained_weight:
            if isinstance(load_pretrained_weight, bool):
                load_pretrained_weight = self.__class__.__name__ + '.h5'

            if not os.path.exists(load_pretrained_weight):
                download(self.WEIGHT_URL, load_pretrained_weight)

            self._model.load(load_pretrained_weight)
            self._model.fc1.params = {}
            self._model.fc2.params = {}
            self._model.fc3.params = {}


class CNN_VGG19(rm.Model):

    def __init__(self, num_class):
        self.block1 = layer_factory(channel=64, conv_layer_num=2)
        self.block2 = layer_factory(channel=128, conv_layer_num=2)
        self.block3 = layer_factory(channel=256, conv_layer_num=4)
        self.block4 = layer_factory(channel=512, conv_layer_num=4)
        self.block5 = layer_factory(channel=512, conv_layer_num=4)
        self.fc1 = rm.Dense(4096)
        self.dropout1 = rm.Dropout(dropout_ratio=0.5)
        self.fc2 = rm.Dense(4096)
        self.dropout2 = rm.Dropout(dropout_ratio=0.5)
        self.fc3 = rm.Dense(num_class)

    def forward(self, x):
        t = self.block1(x)
        t = self.block2(t)
        t = self.block3(t)
        t = self.block4(t)
        t = self.block5(t)
        t = rm.flatten(t)
        t = rm.relu(self.fc1(t))
        t = self.dropout1(t)
        t = rm.relu(self.fc2(t))
        t = self.dropout2(t)
        t = self.fc3(t)
        return t


class CNN_VGG16(rm.Model):

    def __init__(self, num_class):
        self.block1 = layer_factory(channel=64, conv_layer_num=2)
        self.block2 = layer_factory(channel=128, conv_layer_num=2)
        self.block3 = layer_factory(channel=256, conv_layer_num=3)
        self.block4 = layer_factory(channel=512, conv_layer_num=3)
        self.block5 = layer_factory(channel=512, conv_layer_num=3)
        self.fc1 = rm.Dense(4096)
        self.dropout1 = rm.Dropout(dropout_ratio=0.5)
        self.fc2 = rm.Dense(4096)
        self.dropout2 = rm.Dropout(dropout_ratio=0.5)
        self.fc3 = rm.Dense(num_class)

    def forward(self, x):
        t = self.block1(x)
        t = self.block2(t)
        t = self.block3(t)
        t = self.block4(t)
        t = self.block5(t)
        t = rm.flatten(t)
        t = rm.relu(self.fc1(t))
        t = self.dropout1(t)
        t = rm.relu(self.fc2(t))
        t = self.dropout2(t)
        t = self.fc3(t)
        return t


class CNN_VGG11(rm.Model):

    def __init__(self, num_class):
        self.block1 = layer_factory(channel=64, conv_layer_num=1)
        self.block2 = layer_factory(channel=128, conv_layer_num=1)
        self.block3 = layer_factory(channel=256, conv_layer_num=2)
        self.block4 = layer_factory(channel=512, conv_layer_num=2)
        self.block5 = layer_factory(channel=512, conv_layer_num=2)
        self.fc1 = rm.Dense(4096)
        self.dropout1 = rm.Dropout(dropout_ratio=0.5)
        self.fc2 = rm.Dense(4096)
        self.dropout2 = rm.Dropout(dropout_ratio=0.5)
        self.fc3 = rm.Dense(num_class)

    def forward(self, x):
        t = self.block1(x)
        t = self.block2(t)
        t = self.block3(t)
        t = self.block4(t)
        t = self.block5(t)
        t = rm.flatten(t)
        t = rm.relu(self.fc1(t))
        t = self.dropout1(t)
        t = rm.relu(self.fc2(t))
        t = self.dropout2(t)
        t = self.fc3(t)
        return t
