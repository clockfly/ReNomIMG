export default class Model {
  constructor(model_id, project_id, hyper_parameters, algorithm, algorithm_params, state, best_epoch_validation_result, last_epoch, last_batch, total_batch, last_train_loss, running_state) {
    this.model_id = model_id;
    this.project_id = project_id;
    this.hyper_parameters = hyper_parameters;
    this.algorithm = algorithm;
    this.algorithm_params = algorithm_params;
    this.state = state;

    this.train_loss_list = [];
    this.validation_loss_list = [];

    this.best_epoch = undefined;
    this.best_epoch_iou = undefined;
    this.best_epoch_map = undefined;
    this.best_epoch_validation_result = best_epoch_validation_result;

    this.last_epoch = last_epoch;

    // running information
    this.last_batch = last_batch
    this.total_batch = total_batch
    this.last_train_loss = last_train_loss
    this.running_state = running_state
  }
}