import os
import time
import numpy as np
import traceback
import csv
import xml.etree.ElementTree as et
from PIL import Image
from threading import Event
from renom.cuda import set_cuda_active, release_mem_pool

from renom_img.api.detection.yolo_v1 import Yolov1
from renom_img.api.detection.yolo_v2 import Yolov2
from renom_img.api.detection.ssd import SSD
from renom_img.api.utility.load import parse_xml_detection
from renom_img.api.utility.distributor.distributor import ImageDistributor
from renom_img.api.utility.augmentation.process import Shift

from renom_img.server.utility.storage import storage


class PredictionThread(object):

    def __init__(self, thread_id, model_id, hyper_parameters,
                 algorithm, algorithm_params, weight_name, class_map):

        self.model_id = model_id

        # State of thread.
        # The variable _running_state has setter and getter.
        self._running_state = RUN_STATE_STARTING
        self.nth_batch = 0
        self.total_batch = 0
        self.last_batch_loss = 0

        # Error message caused in thread.
        self.error_msg = None

        # hyperparameters
        self.batch_size = int(hyper_parameters["batch_size"])
        self.imsize = (int(hyper_parameters["image_width"]),
                       int(hyper_parameters["image_height"]))
        self.stop_event = Event()

        # Prepare dataset
        self.predict_files = [os.path.join(DATASRC_PREDICTION_IMG, path)
                              for path in sorted(os.listdir(DATASRC_PREDICTION_IMG))]

        # File name of trainded model's weight
        self.weight_name = weight_name

        # Algorithm
        # Pretrained weights are must be prepared.
        self.algorithm = algorithm
        path = os.path.join(DB_DIR_TRAINED_WEIGHT,  self.weight_name)
        if algorithm == ALG_YOLOV1:
            # Algorithm params are saved to h5.
            cell_size = int(algorithm_params["cells"])
            num_bbox = int(algorithm_params["bounding_box"])
            self.model = Yolov1(class_map, cell_size, num_bbox, imsize=self.imsize)
            self.model.load(path)
        elif algorithm == ALG_YOLOV2:
            self.model = Yolov2(class_map, [], imsize=self.imsize)
            self.model.load(path)
        elif algorithm == ALG_SSD:
            self.model = SSD(class_map, imsize=self.imsize)
            self.model.load(path)
        else:
            self.error_msg = "{} is not supported algorithm id.".format(algorithm)

        # Result set of prediction
        self.predict_results = {}

        # File name of prediction result
        self.csv_filename = ''

    @property
    def running_state(self):
        return self._running_state

    @running_state.setter
    def running_state(self, state):
        """
        If thread's state becomes RUN_STATE_STOPPING once, 
        state will never be changed.
        """
        if self._running_state != RUN_STATE_STOPPING:
            self._running_state = state

    def __call__(self):
        # This func works as thread.
        batch_size = self.batch_size

        try:
            i = 0
            set_cuda_active(True)
            release_mem_pool()
            self.model.set_models(inference=True)
            result = []
            # Prediction
            self.running_state = RUN_STATE_PREDICTING
            if self.is_stopped():
                return
            display_loss = 0
            self.total_batch = int(np.ceil(len(self.predict_files) / float(self.batch_size)))
            for i in range(0, self.total_batch):
                self.nth_batch = i
                batch = self.predict_files[i * self.batch_size:(i + 1) * batch_size]
                batch_result = self.model.predict(batch)
                result.extend(batch_result)

            # Set result.
            predict_list = []
            for img_path in self.predict_files:
                img = Image.open(img_path)
                height = img.size[1]
                width = img.size[0]
                predict_list.append({
                    "path": img_path,
                    "height": height,
                    "width": width,
                })

            self.predict_results = {
                "prediction_file_list": predict_list,
                "bbox_list": result
            }

            self.save_predict_result_to_csv()
            self.save_predict_result_to_xml()
        # Store epoch data tp DB.

        except Exception as e:
            traceback.print_exc()
            self.error_msg = str(e)
            self.model = None
            release_mem_pool()

    def get_running_info(self):
        return {
            "model_id": self.model_id,
            "total_batch": self.total_batch,
            "nth_batch": self.nth_batch,
            "last_batch_loss": self.last_batch_loss,
            "running_state": self.running_state,
        }

    def stop(self):
        # Thread can be canceled only if it have not been started.
        # This method is for stopping running thread.
        self.stop_event.set()
        self.running_state = RUN_STATE_STOPPING

    def is_stopped(self):
        return self.stop_event.is_set()

    def save_predict_result_to_csv(self):
        try:
            # modelのcsvを保存する
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            CSV_DIR = os.path.join(BASE_DIR, DATASRC_PREDICTION_OUT, 'csv')
            if not os.path.isdir(CSV_DIR):
                os.makedirs(CSV_DIR)

            self.csv_filename = '{}.csv'.format(int(time.time()))
            filepath = os.path.join(CSV_DIR, self.csv_filename)
            with open(filepath, 'w') as f:
                writer = csv.writer(f, lineterminator="\n")
                for i in range(len(self.predict_results["prediction_file_list"])):
                    row = []
                    row.append(self.predict_results["prediction_file_list"][i]["path"])
                    if isinstance(self.predict_results, list) and len(self.predict_results["bbox_list"]) != 0:
                        for j in range(len(self.predict_results["bbox_list"][i])):
                            b = self.predict_results["bbox_list"][i][j]
                            row.append(b["class"])
                            row.append(b["box"])
                    else:
                        row.append(None)
                        row.append(None)

                    writer.writerow(row)
        except Exception as e:
            traceback.print_exc()
            self.error_msg = e.args[0]

    def save_predict_result_to_xml(self):
        try:
            for i in range(len(self.predict_results["prediction_file_list"])):
                img_path = self.predict_results["prediction_file_list"][i]["path"]
                filename = img_path.split("/")[-1]
                xml_filename = '{}.xml'.format(filename.split(".")[0])
                filepath = os.path.join(DATASRC_PREDICTION_OUT_XML, xml_filename)
                img_path = os.path.join(DATASRC_PREDICTION_IMG, filename)
                im = Image.open(img_path)
                width, height = im.size

                annotation = et.Element("annotation")
                name = et.SubElement(annotation, "filename")
                name.text = filename

                for j in range(len(self.predict_results["bbox_list"][i])):
                    b = self.predict_results["bbox_list"][i][j]
                    obj = et.SubElement(annotation, "object")
                    bbox_class = et.SubElement(obj, "class")
                    bbox_class.text = str(b["class"])

                    bbox_position = et.SubElement(obj, "bndbox")
                    bbox_xmax = et.SubElement(bbox_position, "xmax")
                    xmax = width * b["box"][0] + (width * b["box"][2] / 2.)
                    bbox_xmax.text = str(xmax)

                    bbox_xmin = et.SubElement(bbox_position, "xmin")
                    xmin = width * b["box"][0] - (width * b["box"][2] / 2.)
                    bbox_xmin.text = str(xmin)

                    bbox_ymax = et.SubElement(bbox_position, "ymax")
                    ymax = height * b["box"][1] + height * b["box"][3] / 2.
                    bbox_ymax.text = str(ymax)

                    bbox_ymin = et.SubElement(bbox_position, "ymin")
                    ymin = height * b["box"][1] - height * b["box"][3] / 2.
                    bbox_ymin.text = str(ymin)

                    bbox_score = et.SubElement(obj, "score")
                    bbox_score.text = str(b["score"])

                size = et.SubElement(annotation, "size")
                size_h = et.SubElement(size, "height")
                size_h.text = str(height)
                size_w = et.SubElement(size, "width")
                size_w.text = str(width)

                tree = et.ElementTree(annotation)

                tree.write(filepath)
        except Exception as e:
            traceback.print_exc()
            self.error_msg = e.args[0]
