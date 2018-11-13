export const TASK_ID = {
  CLASSIFICATION: 0,
  DETECTION: 1,
  SEGMENTATION: 2,
}

export const PAGE_ID = {
  DEBUG: -1,
  TRAIN: 0,
  PREDICT: 1,
  DATASET: 2,
}

const COMMON_PARAMS = {
  TRAIN_WHOLE: {
    title: 'Train Whole Network',
    key: 'train_whole', // Must be same as back-end.
    type: 'checkbox',
    default: false,
  },
  BATCH_SIZE: {
    title: 'Batch Size',
    key: 'batch_size',
    type: 'number',
    default: 16,
    min: 4,
    max: 128,
  },
  TOTAL_EPOCH: {
    title: 'Total Epoch',
    key: 'total_epoch',
    disabled: false,
    type: 'number',
    default: 160,
  },
  IMAGE_WIDTH: {
    title: 'Image Width',
    key: 'imsize_w',
    type: 'number',
    disabled: false,
    default: 224,
    min: 64,
    max: 512,
  },
  IMAGE_HEIGHT: {
    title: 'Image Height',
    key: 'imsize_h',
    type: 'number',
    disabled: false,
    default: 224,
    min: 64,
    max: 512,
  },
  LOAD_PRETRAINED_WEIGHT: {
    title: 'Load pretrain weight',
    key: 'load_pretrained_weight',
    disabled: false,
    type: 'checkbox'
  }
}

function override (key, params) {
  return {[key]: {...Object.assign({...COMMON_PARAMS[key]}, params)}}
}

export const ALGORITHM = {
  CLASSIFICATION: {
    ResNet: {
      id: 0,
      key: 'resnet',
      title: 'ResNet',
      params: {
        ...COMMON_PARAMS
      }
    },
    ResNet18: {
      id: 1,
      key: 'resnet18',
      title: 'ResNet18',
      params: {
        ...COMMON_PARAMS,
        PLATEAU: {
          title: 'Plateau',
          key: 'plateau',
          type: 'checkbox',
          default: false
        }
      }
    },
    ResNet34: {
      id: 2,
      key: 'resnet34',
      title: 'ResNet34',
      params: {
        ...COMMON_PARAMS,
        PLATEAU: {
          title: 'Plateau',
          key: 'plateau',
          type: 'checkbox',
          default: false
        }
      }
    },
    ResNet50: {
      id: 3,
      key: 'resnet',
      title: 'ResNet50',
      params: {
        ...COMMON_PARAMS,
        PLATEAU: {
          title: 'Plateau',
          key: 'plateau',
          type: 'checkbox',
          default: false
        }
      }
    },
    ResNet101: {
      id: 4,
      key: 'resnet',
      title: 'ResNet101',
      params: {
        ...COMMON_PARAMS,
        PLATEAU: {
          title: 'Plateau',
          key: 'plateau',
          type: 'checkbox',
          default: false
        }
      }
    },
    ResNet152: {
      id: 5,
      key: 'resnet',
      title: 'ResNet152',
      params: {
        ...COMMON_PARAMS,
        PLATEAU: {
          title: 'Plateau',
          key: 'plateau',
          type: 'checkbox',
          default: false
        }
      }
    },
    DenseNet121: {
      id: 6,
      key: 'densenet121',
      title: 'DenseNet121',
      params: {
        ...COMMON_PARAMS,
      }
    },
    DenseNet169: {
      id: 7,
      key: 'densenet169',
      title: 'DenseNet169',
      params: {
        ...COMMON_PARAMS,
      }
    },
    DenseNet201: {
      id: 8,
      key: 'densenet121',
      title: 'DenseNet121',
      params: {
        ...COMMON_PARAMS,
      }
    },
    VGG11: {
      id: 9,
      key: 'Vgg11',
      title: 'VGG11',
      params: {
        ...COMMON_PARAMS,
      }
    },
    Vgg16: 3,
    Vgg19: 4,
    Inception1: 5,
    Inception2: 6,
    Inception3: 7,
    Inception4: 8,
  },
  DETECTION: {
    YOLOv1: {
      id: 10, // Must be same as back-end.
      key: 'yolov1', // Must be same as back-end.
      title: 'Yolo v1',
      params: {
        ...COMMON_PARAMS,
        ...override('IMAGE_WIDTH', {
          default: 224,
        }),
        ...override('IMAGE_HEIGHT', {
          default: 224,
        }),
        CELL: {
          title: 'Num Cell',
          key: 'cell',
          disabled: false,
          type: 'number',
          default: 7,
        },
        BOX: {
          title: 'Num Box',
          key: 'box',
          disabled: false,
          type: 'number',
          default: 2,
        },
      }
    },
    YOLOv2: {
      id: 11,
      key: 'yolov2',
      title: 'Yolo v2',
      params: {
        ...COMMON_PARAMS,
        ...override('IMAGE_WIDTH', {
          disabled: true,
          default: 320,
        }),
        ...override('IMAGE_HEIGHT', {
          disabled: true,
          default: 320,
        }),
        ANCHOR: {
          title: 'Anchor',
          key: 'anchor',
          type: 'number',
          default: 5,
        },
      }
    },
    SSD: {
      id: 12,
      key: 'ssd',
      title: 'SSD',
      params: {
        ...COMMON_PARAMS,
        ...override('IMAGE_WIDTH', {
          disabled: true,
          default: 300,
        }),
        ...override('IMAGE_HEIGHT', {
          disabled: true,
          default: 300,
        })
      }
    },
  },
  SEGMENTATION: {
    Unet: {
      key: 'unet',
    },
    Fcn: 22,
    TernousNet: 23,
  }
}

export const SORTBY = {
  CLASSIFICATION: {
    MODEL_ID: {
      id: 0,
      key: 'model_id',
      title: 'Model ID',
    },
    VALID_RECALL: {
      id: 1,
      key: 'valid_recall',
      title: 'Valid Recall'
    },
    VALID_PRECISION: {
      id: 2,
      key: 'valid_precision',
      title: 'Valid Precision'
    },
    VALID_F1: {
      id: 3,
      key: 'valid_f1',
      title: 'Valid F1'
    },
    VALID_LOSS: {
      id: 4,
      key: 'valid_loss',
      title: 'Valid Loss'
    }
  },
  DETECTION: {
    MODEL_ID: {
      id: 10,
      key: 'model_id',
      title: 'Model ID',
    },
    VALID_MAP: {
      id: 11,
      key: 'valid_mAP',
      title: 'Valid mAP'
    },
    VALID_IOU: {
      id: 12,
      key: 'valid_iou',
      title: 'Valid IOU'
    },
    VALID_LOSS: {
      id: 13,
      key: 'valid_loss',
      title: 'Valid Loss'
    }

  },
  SEGMENTATION: {
  },
}

export const STATE = {
  CREATED: 0,
  RESERVED: 1,
  STARTED: 2,
  STOPPED: 3,
}

export const RUNNING_STATE = {
  STARTING: 0,
  TRAINING: 1,
  VALIDATING: 2,
  STOPPING: 3,
}

export const FILTER = {
  CLASSIFICATION: {
    VALID_RECALL: {
      key: 'valid_recall',
      title: 'Valid Recall',
      type: 'condition'
    },
    VALID_PRECISION: {
      key: 'valid_precision',
      title: 'Valid Precision',
      type: 'condition'
    },
    VALID_F1: {
      key: 'valid_precision',
      title: 'Valid Precision',
      type: 'condition'
    },
    VALID_LOSS: {
      key: 'valid_precision',
      title: 'Valid Precision',
      type: 'CONDITION'
    },
    ALGORITHM_NAME: {
      key: 'algorithm',
      title: 'Algorithm',
      type: 'SELECT_ALGORITHM'
    }
  },
}

export const FILTER_CONDITION = {
  CONDITION: {
    LESS_THAN: 0,
    EQUAL: 1,
    GRATER_THAN: 2,
  },
}

export function getKeyByValue (object, value) {
  return Object.keys(object).find(key => object[key] === value)
}

export function getKeyByValueIncludes (object, value) {
  return Object.keys(object).find(key => (Object.values(object[key]).some(v => value === v)))
}
