<template>
  <div id="progress-bar">
    <div id="model-id-area">
      <span v-if="isTitle">Model</span>
      <span v-else>{{ model_id }}</span>
    </div>
    <div id="epoch-area">
      <span v-if="isTitle">Epoch</span>
      <span v-else>{{ current_epoch }} / {{ total_epoch }}</span>
    </div>
    <div id="batch-area">
      <span v-if="isTitle">Batch</span>
      <span v-else>{{ current_batch }} / {{ total_batch }}</span>
    </div>
    <div id="loss-area">
      <span v-if="isTitle">Loss</span>
      <span v-else-if="model.isTraining()">{{ loss }}</span>
      <span v-else-if="model.isValidating()">Validating</span>
      <span v-else-if="model.isStopping()">Stopping</span>
      <span v-else-if="model.isWeightDownloading()">Weight Downloading</span>
    </div>
    <div id="bar-area">
      <span v-if="isTitle"/>
      <div
        v-else
        id="bar-background">
        <div
          id="bar-front"
          :style="getWidthOfBar"
          :class="[getColorClass(model), getBarClass]"/>
      </div>
    </div>
    <div
      v-if="!isTitle"
      id="button-stop-area">
      <i
        class="fa fa-stop-circle-o"
        aria-hidden="true"
        @click="onStop"/>
    </div>
  </div>
</template>

<script>
import { mapGetters, mapMutations, mapActions } from 'vuex'
export default {
  name: 'ProgressBar',
  props: {
    model: {
      type: Object,
      default: undefined
    },
    isTitle: {
      type: Boolean,
      default: false
    }
  },
  data: function () {
    return {

    }
  },
  computed: {
    ...mapGetters([
      'getColorClass'
    ]),
    model_id: function () {
      if (this.model === undefined) {
        return '-'
      } else {
        return this.model.id
      }
    },
    getBarClass: function () {
      if (this.model.isValidating() || this.model.isStopping() || this.model.isWeightDownloading()) {
        return 'validating'
      } else {
        return 'training'
      }
    },
    getWidthOfBar: function () {
      if (this.model.isValidating() || this.model.isStopping() || this.model.isWeightDownloading()) {
        return {
          width: '20%'
        }
      } else {
        if (this.total_batch === 0) {
          return {
            width: 0 + '%'
          }
        } else {
          return {
            width: (this.current_batch / this.total_batch) * 100 + '%'
          }
        }
      }
    },
    current_epoch: function () {
      if (this.model === undefined) {
        return '-'
      } else {
        return this.model.nth_epoch
      }
    },
    current_batch: function () {
      if (this.model === undefined) {
        return '-'
      } else {
        return this.model.nth_batch
      }
    },
    total_epoch: function () {
      if (this.model === undefined) {
        return '-'
      } else {
        return this.model.total_epoch
      }
    },
    total_batch: function () {
      if (this.model === undefined) {
        return '-'
      } else {
        return this.model.total_batch
      }
    },
    loss: function () {
      if (this.model === undefined) {
        return '-'
      } else {
        return this.model.last_batch_loss.toFixed(3)
      }
    }
  },
  created: function () {

  },
  methods: {
    ...mapActions(['stopModelTrain']),
    ...mapMutations([
      'showConfirm'
    ]),
    onStop: function () {
      if (this.model) {
        const id = this.model.id
        const func = this.stopModelTrain
        this.showConfirm({
          message: "Are you sure to <span style='color: #f00;}'>stop</span> this model?",
          callback: function () { func(id) }
        })
      }
    }
  }
}
</script>

<style lang='scss'>
#progress-bar {
  display: flex;
  flex-direction: row;
  align-items: center;
  width: 100%;
  height: $progress-bar-height;
  margin-bottom: $progress-bar-margin-bottom;
  font-size: $component-font-size-small;

  #model-id-area {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 12.5%;
    height: 100%;
  }
  #epoch-area {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 12.5%;
    height: 100%;
  }
  #batch-area {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 15%;
    height: 100%;
  }
  #loss-area {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 18%;
    height: 100%;
  }

  #bar-area {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 33%;
    height: 70%;
    #bar-background {
      width: 100%;
      height: calc(100% - #{$progress-bar-margin}*2);
      background-color: $progress-bar-background-color;
      #bar-front.training {
        position: relative;
        top: 0;
        left: 0;
        height: 100%;
        transition: width 300ms;
      }
      #bar-front.validating {
        position: relative;
        top: 0;
        left: 0;
        height: 100%;
        // transition: width 300ms;

        animation: move-bar 1.5s;
        animation-iteration-count: infinite;
        animation-timing-function: linear;
        animation-fill-mode: both;
        animation-delay: 0.1s;
      }

      @keyframes move-bar {
        0% {
          transform: translateX(-50%) scaleX(0);
        }
        20% {
          transform: translateX(0%) scaleX(1);
        }
        80% {
          transform: translateX(400%) scaleX(1);
        }
        100% {
          transform: translateX(450%) scaleX(0);
        }
      }
    }
  }
  #button-stop-area {
    width: 9%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    color: $progress-bar-stop-button-color;
    font-size: $progress-bar-stop-button-font-size;
    i {
      cursor: pointer;
      &:hover {
        color: $progress-bar-stop-button-color-hover;
      }
    }
  }
}
</style>
