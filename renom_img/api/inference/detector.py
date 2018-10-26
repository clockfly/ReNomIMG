import asyncio
import requests
import urllib.request
import numpy as np
import os
from renom_img.server import ALG_YOLOV1, ALG_YOLOV2, ALG_SSD
from renom_img.api.utility.misc.download import download
from renom_img.api.detection.yolo_v1 import Yolov1
from renom_img.api.detection.yolo_v2 import Yolov2
from renom_img.api.detection.ssd import SSD


class Detector(object):
    """This class allows you to pull model which trained on ReNomIMG GUI Tool.

    Args:
        url (string): The url ReNomIMG server running.
        port (string): The port number ReNomIMG server running.
    """

    def __init__(self, url="http://localhost", port='8080'):
        self._pj_id = 1
        self._url = url
        self._port = port
        self._model = None
        self._alg_name = None
        self._model_info = {}

    def __call__(self, x):
        assert self._model is not None, "Please pull trained weight first."
        return self._model(x)

    def pull(self):
        """Pull trained weight from ReNomIMG server.
        Trained weight will be downloaded into current directory.

        Example:
            >>> from renom_img.api.inference.detector import Detector
            >>> detector = Detector()
            >>> detector.pull()

        """
        url = self._url + ':' + self._port
        download_weight_api = "/api/renom_img/v1/projects/{}/deployed_model".format(self._pj_id)
        download_param_api = "/api/renom_img/v1/projects/{}/deployed_model_info".format(self._pj_id)
        download_weight_api = url + download_weight_api
        download_param_api = url + download_param_api

        ret = requests.get(download_param_api).json()
        filename = ret["filename"]

        if not os.path.exists(filename):
            download(download_weight_api, filename)

        if ret["algorithm"] == ALG_YOLOV1:
            self._alg_name = "Yolov1"
            img_w = ret["hyper_parameters"]["image_width"]
            img_h = ret["hyper_parameters"]["image_height"]
            self._model = Yolov1(imsize=(img_h, img_w))
            self._model.load(filename)

        elif ret["algorithm"] == ALG_YOLOV2:
            self._alg_name = "Yolov2"
            img_w = ret["hyper_parameters"]["image_width"]
            img_h = ret["hyper_parameters"]["image_height"]
            self._model = Yolov2(imsize=(img_h, img_w))
            self._model.load(filename)

        elif ret["algorithm"] == ALG_SSD:
            self._alg_name = "SSD"
            img_w = ret["hyper_parameters"]["image_width"]
            img_h = ret["hyper_parameters"]["image_height"]
            self._model = SSD(imsize=(img_h, img_w))
            self._model.load(filename)
            self._model._network.num_class = self._model.num_class

        self._model_info = {
            "Algorithm": self._alg_name,
            "Image size": "{}x{}".format(img_w, img_h),
            "Num class": "{}".format(self._model.num_class)
        }

    def predict(self, img_list):
        """
        Perform prediction to given image.

        Args:
            img_list (string, list, ndarray): Path to the image, list of path or ndarray can be passed.

        Example:
            >>> from renom_img.api.inference.detector import Detector
            >>> detector = Detector()
            >>> detector.pull()
            >>> detector.predict(path_to_image)
            {
              {'box':[0.2, 0.1, 0.5, 0.3], 'class':0, 'name': 'dog', 'score':0.5}
            }
        """
        assert self._model
        return self._model.predict(img_list)

    @property
    def model_info(self):
        """This function returns information of pulled model.

        Example:
            >>> from renom_img.api.inference.detector import Detector
            >>> detector = Detector()
            >>> detector.pull()
            >>> print(detector.model_info)
        """
        return self._model_info
