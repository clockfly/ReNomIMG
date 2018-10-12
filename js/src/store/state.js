export default {
  /*
  header
  */
  // page name
  page_name: '',

  // show nav bar or not
  navigation_bar_shown_flag: false,

  // project data
  project: undefined,
  gpu_num: 1,

  // selected model id
  selected_model_id: undefined,

  // models
  models: [],

  /*
  model list
  */
  // show model add modal or not
  add_model_modal_show_flag: false,

  /*
  alert modal
  */
  // show alert message modal or not
  alert_modal_flag: false,

  // error msg from server
  error_msg: '',

  /*
  tag list
  */
  class_names: {},

  /*
  model sample
  */
  // prediction sample of detection page
  validation_page: 0,
  /*
  Show sample images with modal window
  */
  show_modal_image_sample: false,
  idx_active_image_sample: 0,

  /*
  prediction page
  */
  // predicted result
  predict_results: {'bbox_list': [], 'prediction_file_list': []},
  // predicted csv file name
  csv: '',

  // prediction progress
  predict_running_flag: false,
  predict_total_batch: 0,
  predict_last_batch: 0,

  // predict page
  predict_page: 0,
  predict_page_image_count: 10,

  // show image modal flag
  image_modal_show_flag: false,
  // image data on modal
  image_index_on_modal: undefined,

  /*
  weight
  */
  // weight_exists or not on server
  weight_exists: false,

  // weight downloading progress
  weight_downloading_progress: 0,

  // show weight downloading modal
  weight_downloading_modal: false,

  // dataset defs
  dataset_defs: [],

  // show dataset creating modal
  dataset_creating_modal: false,

  // dataset splitting loading flag
  loading_flg: false,

  // dataset saving flag
  dataset_saving_flg: false,

  // dataset detail info
  dataset_detail: [],

  // modal tabs show flag
  modal_tab_show_flag: true

}
