import os
import sys
import numpy as np
import renom as rm
from tqdm import tqdm

from renom_img.api import adddoc
from renom_img.api.utility.misc.download import download
from renom_img.api.utility.distributor.distributor import ImageDistributor
from renom_img.api.utility.load import load_img
from renom_img.api.utility.target import DataBuilderSegmentation
from renom_img.api.segmentation import SemanticSegmentation

DIR = os.path.split(os.path.abspath(__file__))[0]


def layer_factory(channel=32, conv_layer_num=2):
    layers = []
    for _ in range(conv_layer_num):
        layers.append(rm.Conv2d(channel=channel, padding=1, filter=3))
        layers.append(rm.Relu())
    return rm.Sequential(layers)


@adddoc
class TernausNet(SemanticSegmentation):
    """ TernausNet: U-Net with VGG11 Encoder Pre-Trained on ImageNet for Image Segmentation

    Args:
        class_map(array): Array of class names
        imsize(int or tuple): Input image size
        load_pretrained_weight(bool, str): True if pre-trained weight is used, otherwise False.
        train_whole_network(bool): True if the overall model is trained, otherwise False

    Example:
        >>> import renom as rm
        >>> import numpy as np
        >>> from renom_img.api.segmentation.ternausnet import TernausNet
        >>> n, c, h, w = (2, 12, 64, 64)
        >>> x = rm.Variable(np.random.rand(n, c, h, w))
        >>> model = TernausNet()
        >>> t = model(x)
        >>> t.shape
        (2, 12, 64, 64)

    References:
        | Vladimir Iglovikov, Alexey Shvets
        | **TernausNet: U-Net with VGG11 Encoder Pre-Trained on ImageNet for Image Segmentation**
        | https://arxiv.org/pdf/1801.05746.pdf
        |

    """

    def __init__(self, class_map=[], imsize=(512, 512), load_pretrained_weight=False, train_whole_network=False):
        assert not load_pretrained_weight, "Currently pretrained weight of %s is not prepared. Please set False to `load_pretrained_weight` flag." % self.__class__.__name__
        if not hasattr(imsize, "__getitem__"):
            imsize = (imsize, imsize)
        self.imsize = imsize
        self.num_class = len(class_map)
        self.class_map = [c.encode("ascii", "ignore") for c in class_map]
        self._model = CNN_TernausNet(self.num_class)
        self._train_whole_network = train_whole_network
        self._opt = rm.Sgd(4e-3, 0.9)
        self.decay_rate = 0.00002
        self._freeze()

    def preprocess(self, x):
        """Image preprocess for U-Net.

        :math:`new_x = x/255.`

        Returns:
            (ndarray): Preprocessed data.
        """
        x[:, 0, :, :] -= 123.68  # R
        x[:, 1, :, :] -= 116.779  # G
        x[:, 2, :, :] -= 103.939  # B
        return x

    def get_optimizer(self, current_loss=None, current_epoch=None, total_epoch=None, current_batch=None, total_batch=None):
        if any([num is None for num in
                [current_loss, current_epoch, total_epoch, current_batch, total_batch]]):
            return self._opt
        else:
            ind1 = int(total_epoch * 0.3)
            ind2 = int(total_epoch * 0.5)
            ind3 = int(total_epoch * 0.8)
            if current_epoch == ind1:
                self._opt._lr = 4e-3
            elif current_epoch == ind2:
                self._opt._lr = 1e-3
            elif current_epoch == ind3:
                self._opt.lr = 4e-4
            return self._opt

    def _freeze(self):
        self._model.conv1_1.set_auto_update(self._train_whole_network)
        self._model.conv2_1.set_auto_update(self._train_whole_network)
        self._model.conv3_1.set_auto_update(self._train_whole_network)
        self._model.conv3_2.set_auto_update(self._train_whole_network)
        self._model.conv4_1.set_auto_update(self._train_whole_network)
        self._model.conv4_2.set_auto_update(self._train_whole_network)
        self._model.conv5_1.set_auto_update(self._train_whole_network)
        self._model.conv5_2.set_auto_update(self._train_whole_network)


class CNN_TernausNet(rm.Model):
    def __init__(self, num_class):
        self.conv1_1 = rm.Conv2d(64, padding=1, filter=3)
        self.bn1_1 = rm.BatchNormalize(mode='feature')
        self.conv2_1 = rm.Conv2d(128, padding=1, filter=3)
        self.bn2_1 = rm.BatchNormalize(mode='feature')
        self.conv3_1 = rm.Conv2d(256, padding=1, filter=3)
        self.bn3_1 = rm.BatchNormalize(mode='feature')
        self.conv3_2 = rm.Conv2d(256, padding=1, filter=3)
        self.bn3_2 = rm.BatchNormalize(mode='feature')
        self.conv4_1 = rm.Conv2d(512, padding=1, filter=3)
        self.bn4_1 = rm.BatchNormalize(mode='feature')
        self.conv4_2 = rm.Conv2d(512, padding=1, filter=3)
        self.bn4_2 = rm.BatchNormalize(mode='feature')
        self.conv5_1 = rm.Conv2d(512, padding=1, filter=3)
        self.bn5_1 = rm.BatchNormalize(mode='feature')
        self.conv5_2 = rm.Conv2d(512, padding=1, filter=3)
        self.bn5_2 = rm.BatchNormalize(mode='feature')

        self.deconv1 = rm.Deconv2d(256, stride=2)
        self.conv6 = rm.Conv2d(512, padding=1)
        self.deconv2 = rm.Deconv2d(256, stride=2)
        self.conv7 = rm.Conv2d(512, padding=1)
        self.deconv3 = rm.Deconv2d(128, stride=2)
        self.conv8 = rm.Conv2d(256, padding=1)
        self.deconv4 = rm.Deconv2d(64, stride=2)
        self.conv9 = rm.Conv2d(128, stride=1)
        self.deconv5 = rm.Deconv2d(32, stride=2)
        self.conv10 = rm.Conv2d(num_class, filter=1)

    def forward(self, x):
        c1 = rm.relu(self.bn1_1(self.conv1_1(x)))
        t = rm.max_pool2d(c1, filter=2, stride=2)
        t = rm.dropout(t, 0.5)
        c2 = rm.relu(self.bn2_1(self.conv2_1(t)))
        t = rm.max_pool2d(c2, filter=2, stride=2)
        t = rm.dropout(t, 0.5)
        t = rm.relu(self.bn3_1(self.conv3_1(t)))
        c3 = rm.relu(self.bn3_2(self.conv3_2(t)))
        t = rm.max_pool2d(c3, filter=2, stride=2)
        t = rm.dropout(t, 0.5)
        t = rm.relu(self.bn4_1(self.conv4_1(t)))
        c4 = rm.relu(self.bn4_2(self.conv4_2(t)))
        t = rm.max_pool2d(c4, filter=2, stride=2)
        t = rm.dropout(t, 0.5)
        t = rm.relu(self.bn5_1(self.conv5_1(t)))
        t = rm.relu(self.bn5_2(self.conv5_2(t)))

        t = self.deconv1(t)[:, :, :c4.shape[2], :c4.shape[3]]
        t = rm.concat([c4, t])
        t = rm.relu(self.conv6(t))
        t = self.deconv2(t)[:, :, :c3.shape[2], :c3.shape[3]]
        t = rm.concat([c3, t])

        t = rm.relu(self.conv7(t))
        t = self.deconv3(t)[:, :, :c2.shape[2], :c2.shape[3]]
        t = rm.concat([c2, t])

        t = rm.relu(self.conv8(t))
        t = self.deconv4(t)[:, :, :c1.shape[2], :c1.shape[3]]
        t = rm.concat([c1, t])

        t = rm.relu(self.conv9(t))
        t = self.deconv5(t)[:, :, :c1.shape[2], :c1.shape[3]]
        t = rm.concat([c1, t])

        t = self.conv10(t)
        return t
