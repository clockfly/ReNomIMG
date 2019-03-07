# coding: utf-8

import os
import json
import time
import base64
import threading
import pkg_resources
import argparse
import urllib
import mimetypes
import posixpath
import traceback
import pathlib
import random
from collections import OrderedDict
import xmltodict

import PIL
import numpy as np
from concurrent.futures import ThreadPoolExecutor as Executor
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import CancelledError
from bottle import HTTPResponse, default_app, route, static_file, request, error

from renom.cuda import release_mem_pool
from renom_img.server import create_dirs
create_dirs()

from renom_img.api.utility.load import parse_xml_detection
from renom_img.server import wsgi_server
from renom_img.server.train_thread import TrainThread
from renom_img.server.prediction_thread import PredictionThread
from renom_img.server.utility.storage import storage

# Constants
from renom_img.server import MAX_THREAD_NUM, DB_DIR_TRAINED_WEIGHT, GPU_NUM
from renom_img.server import DATASRC_IMG, DATASRC_LABEL, DATASRC_DIR, DATASRC_PREDICTION_OUT
from renom_img.server import STATE_FINISHED, STATE_RUNNING, STATE_DELETED, STATE_RESERVED
from renom_img.server import WEIGHT_EXISTS, WEIGHT_CHECKING, WEIGHT_DOWNLOADING

executor = Executor(max_workers=MAX_THREAD_NUM)

# Thread(Future object) is stored to thread_pool as pair of "thread_id:[future, thread_obj]".
train_thread_pool = {}
prediction_thread_pool = {}

confirm_dataset = OrderedDict()


def get_train_thread_count():
    return len([th for th in train_thread_pool.values() if th[0].running()])


def create_response(body):
    r = HTTPResponse(status=200, body=body)
    r.set_header('Content-Type', 'application/json')
    return r


def strip_path(filename):
    if os.path.isabs(filename):
        raise ValueError('Invalid path')
    if '..' in filename:
        raise ValueError('Invalid path')
    if ':' in filename:
        raise ValueError('Invalid path')

    filename = filename.strip().strip('./\\')
    return filename


def _get_resource(path, filename):
    filename = strip_path(filename)
    body = pkg_resources.resource_string(__name__, posixpath.join('.build', path, filename))

    headers = {}
    mimetype, encoding = mimetypes.guess_type(filename)
    if mimetype:
        headers['Content-Type'] = mimetype
    if encoding:
        headers['encoding'] = encoding
    return HTTPResponse(body, **headers)


@route("/")
def index():
    return _get_resource('', 'index.html')


@route("/static/<file_name:re:.+>")
def static(file_name):
    return _get_resource('static', file_name)


@route("/css/<file_name:path>")
def css(file_name):
    return _get_resource('static/css/', file_name)


@route("/fonts/<file_name:path>")
def font(file_name):
    return _get_resource('static/fonts/', file_name)


@error(404)
def error404(error):
    print(error)
    body = json.dumps({"error_msg": "Page Not Found"})
    ret = create_response(body)
    return ret


@route("/datasrc/<folder_name:path>/<file_name:path>")
def datasrc(folder_name, file_name):
    file_dir = os.path.join('datasrc', folder_name)
    return static_file(file_name, root=file_dir, mimetype='image/*')


@route("/api/renom_img/v1/projects/<project_id:int>", method="GET")
def get_project(project_id):
    try:
        kwargs = {}
        kwargs["fields"] = "project_id,project_name,project_comment,deploy_model_id"

        data = storage.fetch_project(project_id, **kwargs)
        data['gpu_num'] = GPU_NUM or 1
        body = json.dumps(data)
    except Exception as e:
        traceback.print_exc()
        body = json.dumps({"error_msg": e.args[0]})

    ret = create_response(body)
    return ret


@route("/api/renom_img/v1/projects/<project_id:int>/models/<model_id:int>", method="GET")
def get_model(project_id, model_id):
    try:
        kwargs = {}
        if request.params.fields != '':
            kwargs["fields"] = request.params.fields

        data = storage.fetch_model(project_id, model_id, **kwargs)
        body = json.dumps(data)

    except Exception as e:
        traceback.print_exc()
        body = json.dumps({"error_msg": e.args[0]})

    ret = create_response(body)
    return ret


@route("/api/renom_img/v1/projects/<project_id:int>/models", method="GET")
def get_models(project_id):
    """

    Args:

    Return:
        Success:
            Json string
            {
              :

            }
    """

    try:
        data = storage.fetch_models(project_id)
        body = json.dumps(data)
        ret = create_response(body)
        return ret

    except Exception as e:
        traceback.print_exc()
        body = json.dumps({"error_msg": e.args[0]})
        ret = create_response(body)
        return ret


@route("/api/renom_img/v1/projects/<project_id:int>/model/create", method="POST")
def create_model(project_id):
    try:
        model_id = storage.register_model(
            project_id=project_id,
            dataset_def_id=json.loads(request.params.dataset_def_id),
            hyper_parameters=json.loads(request.params.hyper_parameters),
            algorithm=request.params.algorithm,
            algorithm_params=json.loads(request.params.algorithm_params)
        )

        if get_train_thread_count() >= MAX_THREAD_NUM:
            storage.update_model_state(model_id, STATE_RESERVED)
        else:
            storage.update_model_state(model_id, STATE_RUNNING)

        data = {"model_id": model_id}
        body = json.dumps(data)
    except Exception as e:
        traceback.print_exc()
        body = json.dumps({"error_msg": e.args[0]})
    ret = create_response(body)
    return ret


@route("/api/renom_img/v1/projects/<project_id:int>/models/<model_id:int>/run", method="GET")
def run_model(project_id, model_id):
    """
    Create thread(Future object) and submit it to executor.
    The thread is stored to train_thread_pool as a pair of thread_id and thread.
    """
    try:
        fields = 'hyper_parameters,algorithm,algorithm_params,dataset_def_id'
        data = storage.fetch_model(project_id, model_id, fields=fields)
        thread_id = "{}_{}".format(project_id, model_id)
        th = TrainThread(thread_id, project_id, model_id,
                         data['dataset_def_id'],
                         data["hyper_parameters"],
                         data['algorithm'], data['algorithm_params'])
        ft = executor.submit(th)
        train_thread_pool[thread_id] = [ft, th]

        try:
            # This will wait for end of thread.
            ft.result()
            ft.cancel()
        except CancelledError as ce:
            # If the model is deleted or stopped,
            # program reaches here.
            pass
        error_msg = th.error_msg
        del train_thread_pool[thread_id]
        ft = None
        th = None

        model = storage.fetch_model(project_id, model_id, fields='state')
        if model['state'] != STATE_DELETED:
            storage.update_model_state(model_id, STATE_FINISHED)
        release_mem_pool()

        if error_msg is not None:
            body = json.dumps({"error_msg": error_msg})
            ret = create_response(body)
            return ret
        body = json.dumps({"dummy": ""})
        ret = create_response(body)
        return ret

    except Exception as e:
        release_mem_pool()
        traceback.print_exc()
        body = json.dumps({"error_msg": e.args[0]})
        ret = create_response(body)
        return ret


@route("/api/renom_img/v1/projects/<project_id:int>/models/<model_id:int>/progress", method="POST")
def progress_model(project_id, model_id):
    try:
        try:
            req_last_batch = request.params.get("last_batch", None)
            req_last_batch = int(req_last_batch) if req_last_batch is not None else 0
            req_last_epoch = request.params.get("last_epoch", None)
            req_last_epoch = int(req_last_epoch) if req_last_epoch is not None else 0
            req_running_state = request.params.get("running_state", None)
            req_running_state = int(req_running_state) if req_running_state is not None else 0
        except Exception as e:
            req_last_batch = 0
            req_last_epoch = 0
            req_running_state = 0

        thread_id = "{}_{}".format(project_id, model_id)
        for j in range(1800):
            time.sleep(0.75)
            th = train_thread_pool.get(thread_id, None)
            model_state = storage.fetch_model(project_id, model_id, fields="state")["state"]
            if th is not None:
                th = th[1]
                # If thread status updated, return response.
                if isinstance(th, TrainThread) and th.nth_epoch != req_last_epoch and th.valid_loss_list:
                    try:
                        map_list = np.array(th.valid_map_list)
                        occurences = np.where(map_list == map_list.max())[0]
                        best_epoch = int(occurences[-1])
                        body = {
                            "total_batch": th.total_batch,
                            "last_batch": th.nth_batch,
                            "last_epoch": th.nth_epoch,
                            "batch_loss": th.last_batch_loss,
                            "running_state": th.running_state,
                            "state": model_state,
                            "validation_loss_list": th.valid_loss_list,
                            "train_loss_list": th.train_loss_list,
                            "best_epoch": best_epoch,
                            "best_epoch_iou": th.valid_iou_list[best_epoch],
                            "best_epoch_map": th.valid_map_list[best_epoch],
                            "best_epoch_validation_result": th.valid_predict_box[best_epoch]
                        }
                        body = json.dumps(body)
                        ret = create_response(body)
                        return ret
                    except Exception as e:
                        traceback.print_exc()
                        import pdb
                        pdb.set_trace()

                elif isinstance(th, TrainThread) and (th.nth_batch != req_last_batch or
                                                      th.running_state != req_running_state or
                                                      th.weight_existance == WEIGHT_DOWNLOADING):
                    body = {
                        "total_batch": th.total_batch,
                        "last_batch": th.nth_batch,
                        "last_epoch": th.nth_epoch,
                        "batch_loss": th.last_batch_loss,
                        "running_state": th.running_state,
                        "state": model_state,
                        "validation_loss_list": [],
                        "train_loss_list": [],
                        "best_epoch": 0,
                        "best_epoch_iou": 0,
                        "best_epoch_map": 0,
                        "best_epoch_validation_result": []
                    }
                    body = json.dumps(body)
                    ret = create_response(body)
                    return ret

    except Exception as e:
        traceback.print_exc()
        body = json.dumps({"error_msg": e.args[0]})
        ret = create_response(body)
        return ret


@route("/api/renom_img/v1/projects/<project_id:int>/models/<model_id:int>/stop", method="GET")
def stop_model(project_id, model_id):
    try:
        thread_id = "{}_{}".format(project_id, model_id)

        th = train_thread_pool.get(thread_id, None)
        if th is not None:
            if not th[0].cancel():
                th[1].stop()
                th[0].result()  # Same as join.
            storage.update_model_state(model_id, STATE_FINISHED)

    except Exception as e:
        traceback.print_exc()
        body = json.dumps({"error_msg": e.args[0]})
        ret = create_response(body)
        return ret


@route("/api/renom_img/v1/projects/<project_id:int>/models/<model_id:int>", method="DELETE")
def delete_model(project_id, model_id):
    try:
        thread_id = "{}_{}".format(project_id, model_id)
        storage.update_model_state(model_id, STATE_DELETED)
        th = train_thread_pool.get(thread_id, None)
        if th is not None:
            if not th[0].cancel():
                th[1].stop()
                th[0].result()

        ret = storage.fetch_model(project_id, model_id, "best_epoch_weight")
        file_name = ret.get('best_epoch_weight', None)
        if file_name is not None:
            weight_path = os.path.join(DB_DIR_TRAINED_WEIGHT, file_name)
            if os.path.exists(weight_path):
                os.remove(weight_path)

    except Exception as e:
        traceback.print_exc()
        body = json.dumps({"error_msg": e.args[0]})
        ret = create_response(body)
        return ret


@route("/api/renom_img/v1/projects/<project_id:int>/models/update/state", method="GET")
def update_models_state(project_id):
    try:
        # set running model information for polling
        for _ in range(60):
            body = {}
            models = storage.fetch_models(project_id)
            reserved_count = 0
            running_count = 0
            for k in list(models.keys()):
                model_id = models[k]["model_id"]
                running_state = models[k]["running_state"]
                state = models[k]['state']
                body[model_id] = {
                    'running_state': running_state,
                    'state': state
                }
                if state == STATE_RESERVED:
                    reserved_count += 1
                if state == STATE_RUNNING:
                    running_count += 1

            if reserved_count > 0 and running_count < MAX_THREAD_NUM:
                time.sleep(2)
            else:
                break

        body = json.dumps(body)
        ret = create_response(body)
        return ret
    except Exception as e:
        traceback.print_exc()
        body = json.dumps({"error_msg": e.args[0]})
        ret = create_response(body)
        return ret


@route("/api/renom_img/v1/dataset_defs", method="GET")
def get_datasets():
    try:
        recs = storage.fetch_dataset_defs()
        ret = []
        for rec in recs:
            missing_image_path = ""
            id, name, ratio, description, train_imgs, valid_imgs, class_map, class_tag_list, created, updated = rec
            valid_img_names = [os.path.join("datasrc/img/", path) for path in valid_imgs]
            valid_imgs = []
            for img_name in valid_img_names:
                try:
                    im = PIL.Image.open(img_name)
                    width, height = im.size
                except Exception:
                    missing_image_path = img_name
                    traceback.print_exc()
                    width = height = 50
                    break
                valid_imgs.append(dict(filename=img_name, width=width, height=height))

            if not missing_image_path:
                for img_name in train_imgs:
                    path = os.path.join("datasrc/img/", img_name)
                    if not os.path.exists(path):
                        missing_image_path = path
                        break

            if missing_image_path:
                raise Exception("Image '{}' included in the dataset '{}' does not exist.".format(
                    missing_image_path, name))

            ret.append(dict(id=id,
                            name=name,
                            ratio=ratio,
                            description=description,
                            train_imgs=len(train_imgs),
                            valid_imgs=valid_imgs,
                            class_map=class_map,
                            class_tag_list=class_tag_list,
                            created=created,
                            updated=updated
                            )
                       )
        return create_response(json.dumps({'dataset_defs': ret}))

    except Exception as e:
        traceback.print_exc()
        body = json.dumps({"error_msg": e.args[0]})
        ret = create_response(body)
        return ret


@route("/api/renom_img/v1/load_dataset_split_detail", method=['POST', 'GET'])
def load_dataset_split_detail():
    import time
    try:
        start_t = time.time()
        datasrc = pathlib.Path(DATASRC_DIR)
        imgdirname = pathlib.Path("img")
        xmldirname = pathlib.Path("label")

        imgdir = (datasrc / imgdirname)
        xmldir = (datasrc / xmldirname)

        name = urllib.parse.unquote(request.params.name, encoding='utf-8')
        ratio = float(request.params.ratio)
        client_id = request.params.u_id
        description = urllib.parse.unquote(request.params.description, encoding='utf-8')

        start_t = time.time()

        # if 2nd time delete confirmdataset id
        if request.params.delete_id:
            del confirm_dataset[request.params.delete_id]

        # old cofirm_dataset delete
        if len(confirm_dataset) > 100:
            confirm_dataset.popitem(False)

        # search image files
        imgs = (p.relative_to(imgdir) for p in imgdir.iterdir() if p.is_file())

        # remove images without label
        imgs = set([img for img in imgs if (xmldir / img).with_suffix('.xml').is_file()])
        assert len(imgs) > 0, "Image not found in directory. Please set images to 'datasrc/img' directory and xml files to 'datasrc/label' directory."

        # split files into trains and validations
        n_imgs = len(imgs)

        trains = set(random.sample(imgs, int(ratio * n_imgs)))
        valids = imgs - trains

        start_t = time.time()
        # build filename of images and labels
        train_imgs = [str(img) for img in trains]
        valid_imgs = [str(img) for img in valids]

        perm = np.random.permutation(int(n_imgs))
        perm_train, perm_valid = np.split(perm, [int(n_imgs * ratio)])
        imgs = list(imgs)

        parsed_train_imgs = []
        parsed_valid_imgs = []

        parsed_train_img_names = [str(imgs[perm]).split('.')[0] for perm in perm_train]
        parsed_valid_img_names = [str(imgs[perm]).split('.')[0] for perm in perm_valid]

        start_t = time.time()

        parsed_train, train_class_map = parse_xml_detection([str(path) for path in xmldir.iterdir() if str(
            path).split(os.sep)[-1].split('.')[0] in parsed_train_img_names], num_thread=8)

        parsed_valid, valid_class_map = parse_xml_detection([str(path) for path in xmldir.iterdir() if str(
            path).split(os.sep)[-1].split('.')[0] in parsed_valid_img_names], num_thread=8)

        start_t = time.time()
        # Insert detailed informations
        train_num = len(train_imgs)
        valid_num = len(valid_imgs)
        class_tag_list = []

        train_tag_count = {}
        for i in range(len(parsed_train)):
            for j in range(len(train_class_map)):
                if parsed_train[i][0].get('name') == train_class_map[j]:
                    if train_class_map[j] not in train_tag_count:
                        train_tag_count[train_class_map[j]] = 1
                    else:
                        train_tag_count[train_class_map[j]] += 1

        valid_tag_count = {}
        for i in range(len(parsed_valid)):
            for j in range(len(valid_class_map)):
                if parsed_valid[i][0].get('name') == valid_class_map[j]:
                    if valid_class_map[j] not in valid_tag_count:
                        valid_tag_count[valid_class_map[j]] = 1
                    else:
                        valid_tag_count[valid_class_map[j]] += 1

        for tags in sorted(train_tag_count.keys()):
            class_tag_list.append({
                "tags": tags,
                "train": train_tag_count.get(tags),
                "valid": valid_tag_count.get(tags)
            })

        # save datasplit setting
        confirm_dataset[client_id] = {
            "name": name,
            "ratio": ratio,
            "description": description,
            "train_imgs": train_imgs,
            "valid_imgs": valid_imgs,
            "class_maps": train_class_map,
            "class_tag_list": class_tag_list
        }

        body = json.dumps(
            {"total": n_imgs,
             "id": client_id,
             "description": description,
             "train_image_num": train_num,
             "valid_image_num": valid_num,
             "class_tag_list": class_tag_list,
             "train_imgs": train_imgs,
             "valid_imgs": valid_imgs,
             })

        ret = create_response(body)
        return ret

    except Exception as e:
        traceback.print_exc()
        body = json.dumps({"error_msg": e.args[0]})
        ret = create_response(body)
        return ret


@route("/api/renom_img/v1/weights/progress/<progress_num:int>", method="GET")
def weight_download_progress(progress_num):
    try:
        for i in range(60):
            for th in train_thread_pool.values():
                if isinstance(th, TrainThread) and th[1].weight_existance == WEIGHT_CHECKING:
                    pass
                elif th[1].weight_existance == WEIGHT_EXISTS:
                    body = json.dumps({"progress": 100})
                    ret = create_response(body)
                    return ret
                elif th[1].weight_existance == WEIGHT_DOWNLOADING:
                    if th[1].percentage > 10 * progress_num:
                        body = json.dumps({"progress": th[1].percentage})
                        ret = create_response(body)
                        return ret
            time.sleep(1)

    except Exception as e:
        traceback.print_exc()
        body = json.dumps({"error_msg": e.args[0]})
        ret = create_response(body)
        return ret


@route("/api/renom_img/v1/dataset_defs/", method="POST")
def create_dataset_def():
    try:
        # register dataset
        client_id = request.params.u_id
        name = urllib.parse.unquote(request.params.name, encoding='utf-8')
        description = urllib.parse.unquote(request.params.description, encoding='utf-8')

        id = storage.register_dataset_def(
            name,
            confirm_dataset[client_id].get('ratio'),
            description,
            confirm_dataset[client_id].get('train_imgs'),
            confirm_dataset[client_id].get('valid_imgs'),
            confirm_dataset[client_id].get('class_maps'),
            confirm_dataset[client_id].get('class_tag_list')
        )

        # Insert detailed informations

        del confirm_dataset[client_id]

        body = json.dumps({"id": id})
        ret = create_response(body)
        return ret

    except Exception as e:
        traceback.print_exc()
        body = json.dumps({"error_msg": e.args[0]})
        ret = create_response(body)
        return ret


@route("/api/renom_img/v1/projects/<project_id:int>/models/<model_id:int>/run_prediction", method="GET")
def run_prediction(project_id, model_id):
    # 学習データ読み込み
    try:
        thread_id = "{}_{}".format(project_id, model_id)
        fields = 'hyper_parameters,algorithm,algorithm_params,best_epoch_weight,dataset_def_id'
        data = storage.fetch_model(project_id, model_id, fields=fields)
        (id, name, ratio, train_imgs, valid_imgs, class_map, created,
         updated) = storage.fetch_dataset_def(data['dataset_def_id'])
        # weightのh5ファイルのパスを取得して予測する
        with Executor(max_workers=MAX_THREAD_NUM) as prediction_executor:
            th = PredictionThread(thread_id, model_id, data["hyper_parameters"], data["algorithm"],
                                  data["algorithm_params"], data["best_epoch_weight"], class_map)
            ft = prediction_executor.submit(th)
            prediction_thread_pool[thread_id] = [ft, th]
        ft.result()

        if th.error_msg is not None:
            body = json.dumps({"error_msg": th.error_msg})
        else:
            data = {
                "predict_results": th.predict_results,
                "csv": th.csv_filename,
            }
            body = json.dumps(data)
    except Exception as e:
        traceback.print_exc()
        body = json.dumps({"error_msg": e.args[0]})

    ret = create_response(body)
    return ret


@route("/api/renom_img/v1/projects/<project_id:int>/models/<model_id:int>/prediction_info", method="GET")
def prediction_info(project_id, model_id):
    try:
        # 学習データ読み込み
        thread_id = "{}_{}".format(project_id, model_id)
        while True:
            if thread_id in prediction_thread_pool:
                _, th = prediction_thread_pool[thread_id]
                break
            else:
                time.sleep(1)
        time.sleep(1)
        data = {
            "predict_total_batch": th.total_batch,
            "predict_last_batch": th.nth_batch,
        }
        body = json.dumps(data)
        ret = create_response(body)
        return ret
    except Exception as e:
        traceback.print_exc()
        body = json.dumps({"error_msg": e.args[0]})

    ret = create_response(body)
    return ret


@route("/api/renom_img/v1/projects/<project_id:int>/models/<model_id:int>/export_csv/<filename>", method="GET")
def export_csv(project_id, model_id, filename):
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        csv_dir = os.path.join(BASE_DIR, DATASRC_PREDICTION_OUT, 'csv')
        return static_file(filename, root=csv_dir, download=True)
    except Exception as e:
        traceback.print_exc()
        body = json.dumps({"error_msg": e.args[0]})
        ret = create_response(body)
        return ret


@route("/api/renom_img/v1/projects/<project_id:int>/models/<model_id:int>/deploy", method="GET")
def deploy_model(project_id, model_id):
    try:
        storage.update_project_deploy(project_id, model_id)
    except Exception as e:
        traceback.print_exc()
        body = json.dumps({"error_msg": e.args[0]})
        ret = create_response(body)
        return ret


@route("/api/renom_img/v1/projects/<project_id:int>/models/<model_id:int>/undeploy", method="GET")
def undeploy_model(project_id, model_id):
    try:
        storage.update_project_deploy(project_id, None)
    except Exception as e:
        traceback.print_exc()
        body = json.dumps({"error_msg": e.args[0]})
        ret = create_response(body)
        return ret


@route("/api/renom_img/v1/projects/<project_id:int>/deployed_model", method="GET")
def pull_deployed_model(project_id):
    # This method will be called from python script.
    try:
        deployed_id = storage.fetch_deployed_model_id(project_id)[0]['deploy_model_id']
        ret = storage.fetch_model(project_id, deployed_id, "best_epoch_weight")
        file_name = ret['best_epoch_weight']
        path = DB_DIR_TRAINED_WEIGHT
        return static_file(file_name, root=path, download='deployed_model.h5')
    except Exception as e:
        traceback.print_exc()
        body = json.dumps({"error_msg": e.args[0]})
        ret = create_response(body)
        return ret


@route("/api/renom_img/v1/projects/<project_id:int>/deployed_model_info", method="GET")
def get_deployed_model_info(project_id):
    # This method will be called from python script.
    try:
        deployed_id = storage.fetch_deployed_model_id(project_id)[0]['deploy_model_id']
        ret = storage.fetch_model(project_id, deployed_id, "best_epoch_weight")
        file_name = ret['best_epoch_weight']
        ret = storage.fetch_model(project_id, deployed_id,
                                  "algorithm,algorithm_params,hyper_parameters")
        ret["filename"] = file_name
        body = json.dumps(ret)
        ret = create_response(body)
        return ret
    except Exception as e:
        traceback.print_exc()
        body = json.dumps({"error_msg": e.args[0]})
        ret = create_response(body)
        return ret


@route("/api/renom_img/v1/projects/<project_id:int>/class_map", method="GET")
def get_class_map(project_id):
    try:
        ret = storage.fetch_class_map()
        body = json.dumps({"class_map": ret["class_map"]})
        ret = create_response(body)
        return ret
    except Exception as e:
        body = json.dumps({"error_msg": e.args[0]})
        ret = create_response(body)
        return ret


def main():
    # Creates directory only if server starts.
    create_dirs()
    # Parser settings.
    parser = argparse.ArgumentParser(description='ReNomIMG')
    parser.add_argument('--host', default='0.0.0.0', help='Server address')
    parser.add_argument('--port', default='8080', help='Server port')

    args = parser.parse_args()

    wsgiapp = default_app()
    httpd = wsgi_server.Server(wsgiapp, host=args.host, port=int(args.port))
    httpd.serve_forever()


if __name__ == "__main__":
    main()
