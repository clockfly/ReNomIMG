<template>
  <div id="modal-add-dataset">

    <!--Left page of creating dataset---------->
    <div id="dataset-setting">
      <div
        v-if="isTestDataset"
        id="title"> Test Dataset Setting
      </div>
      <div
        v-else
        id="title"> Dataset Setting
      </div>
      <div
        v-if="false"
        class="item">
        Create as Test Dataset
        <input
          v-model="isTestDataset"
          type="checkbox"
          value="false"
          placeholder="false">
      </div>
      <div class="item">
        Name<input
          ref="nameText"
          v-model="nameText"
          :class="{'show-short-period': notifyNameField}"
          :maxlength="nameMaxLength"
          type="text"
          placeholder="dataset"
          @keydown="nameInputNotify">
      </div>
      <div
        v-if="false"
        class="item">
        Test Dataset
        <select
          v-model="test_dataset"
          :disabled="isTestDataset">
          <option
            disabled
            value=""
            selected>Select Test Dataset</option>
          <option value="none">--none--</option>
          <option
            v-for="(item, key) in getFilteredTestDatasetList"
            :key="key"
            :value="item">{{ item.name }}</option>
        </select>
      </div>
      <div class="item">
        Description<textarea
          ref="descriptionText"
          :class="{'show-short-period': notifyDescriptionField}"
          v-model="descriptionText"
          :maxlength="descriptionMaxLength"
          type="text"
          placeholder="description"
          @keydown="descriptionInputNotify"/>
      </div>
      <div class="item">
        Ratio
        <span
          v-if="ratio < 0.01 || ratio > 0.99"
          class="warn">Ratio must be '0.01 &lt;= Ratio &lt;= 0.99'</span>
        <input
          v-model="ratio"
          type="number"
          placeholder="0.8"
          maxlength="3"
          step="0.1"
          min="0.01"
          max="0.99">
      </div>
      <input
        :disabled="!confirmable"
        type="button"
        value="Confirm"
        @click="onConfirmDataset">
    </div>
    <!----------Left page of creating dataset-->

    <!--Right page of creating dataset---------->
    <div id="dataset-confirm">
      <div id="title">
        Dataset BreakDown
      </div>
      <div id="dataset-params">
        <span> Name : {{ name }} </span>
      </div>
      <div id="dataset-params">
        <span> Ratio : {{ ratio }} </span>
      </div>
      <div id="dataset-numbers">
        <div class="num-item">
          <div class="num-title">
            Total Images :
          </div>
          <div class="num">
            {{ total_num }}
          </div>
        </div>
      </div>

      <div
        id="dataset-ratio-bar"
        :class="{'bar-anime': confirming_dataset}"
        @mouseenter="isHovering=true"
        @mouseleave="isHovering=false">
        <section
          :style="train_num_style"
          class="color-train">
          <span>Train</span>
        </section>
        <section
          :style="valid_num_style"
          class="color-valid">
          <span>Valid</span>
        </section>
      </div>
      <div id="breakdown">
        <div
          v-if="confirming_flag"
          id="load-progress">
          <div class="lds-ripple"><div/><div/></div>
          Loading Dataset...
        </div>
        <div
          v-for="(item, key) in class_items"
          v-else
          id="class-ratio-bars"
          :key="key">
          <span>{{ item[0] }}</span>
          <div
            :class="{'bar-anime': confirming_dataset}"
            :style="{width: item[1] + item[2] + '%'}"
            class="bar">
            <section
              :style="{width: item[1]/(item[1] + item[2])*100 + '%'}"
              class="color-train"/>
            <section
              :style="{width: item[2]/(item[1] + item[2])*100 + '%'}"
              class="color-valid"/>
          </div>
        </div>
      </div>
      <input
        id="submit-button"
        :disabled="!submitable"
        type="button"
        value="submit"
        @click="onAddDataset">
    </div>
    <!----------Right page of creating dataset-->
  </div>
</template>

<script>
import { DATASET_NAME_MAX_LENGTH,
  DATASET_DESCRIPTION_MAX_LENGTH, DATASET_DESCRIPTION_MIN_LENGTH } from '@/const.js'
import { mapGetters, mapMutations, mapState, mapActions } from 'vuex'
import BreakDownBar from '@/components/page/train_page/breakdown_ratio_bar.vue'
import DatasetDetailBar from '@/components/page/train_page/dataset_detail_ratio_bar.vue'

export default {
  name: 'ModalAddDataset',
  components: {
    'dataset-ratio-bar': DatasetDetailBar,
    'breakdown-ratio-bar': BreakDownBar
  },
  data: function () {
    return {
      name: '',
      description: '',
      ratio: 0.8,
      isTestDataset: false,
      test_dataset: '',
      timeStamp: '',
      isHovering: false,
      notifyNameField: false,
      notifyDescriptionField: false,
      nameFieldTimeoutFunc: function () {}
    }
  },
  computed: {
    ...mapState([
      'confirming_flag',
      'confirming_dataset'
    ]),
    ...mapGetters([
      'getFilteredTestDatasetList',
    ]),
    nameMaxLength: function () { return DATASET_NAME_MAX_LENGTH },
    nameMinLength: function () { return DATASET_NAME_MAX_LENGTH },
    descriptionMaxLength: function () { return DATASET_DESCRIPTION_MAX_LENGTH },
    descriptionMinLength: function () { return DATASET_DESCRIPTION_MIN_LENGTH },
    nameText: {
      get () { return this.name },
      set (v) {
        this.name = v
        this.notifyNameField = (this.name.length === this.nameMaxLength && this.notifyNameField)
      }
    },
    descriptionText: {
      get () { return this.description },
      set (v) {
        this.description = v
        this.notifyDescriptionField = (this.description.length === this.descriptionMaxLength && this.notifyDescriptionField)
      }
    },
    confirmable: function () {
      if (!this.name || this.ratio <= 0 || this.ratio >= 1 || this.confirming_flag) {
        return false
      }
      return true
    },
    submitable: function () {
      if (!this.confirming_dataset || this.confirming_flag) {
        return false
      }
      return true
    },
    info: function () {
      const dataset = this.confirming_dataset
      if (!dataset) return
      const info = dataset.class_info
      return info
    },
    total_num: function () {
      const info = this.info
      if (!info) return
      return info.train_img_num + info.valid_img_num
    },
    train_num: function () {
      const info = this.info
      if (!info) return 0
      return info.train_img_num
    },
    valid_num: function () {
      const info = this.info
      if (!info) return 0
      return info.valid_img_num
    },
    class_items: function () {
      const dataset = this.confirming_dataset
      if (!dataset) return
      const class_map = dataset.class_map
      if (!this.info) return
      const train_list = this.info.train_ratio
      const valid_list = this.info.valid_ratio
      const class_list = this.info.class_ratio
      return train_list.map((t, index) => [
        class_map[index],
        (t) * class_list[index] * 100,
        valid_list[index] * class_list[index] * 100
      ])
    },
    train_num_style: function () {
      if (this.total_num <= 0) return
      return {
        width: (this.train_num / this.total_num) * 100 + '%',
      }
    },
    valid_num_style: function () {
      if (this.total_num <= 0) return
      return {
        width: (this.valid_num / this.total_num) * 100 + '%',
      }
    },
  },
  beforeMount: function () {
    this.reset()
  },
  mounted: function () {
    this.$refs.nameText.addEventListener('animationend', () => {
      this.notifyNameField = false
    })
    this.$refs.descriptionText.addEventListener('animationend', () => {
      this.notifyDescriptionField = false
    })
  },
  methods: {
    ...mapMutations([
      'setConfirmingFlag',
      'setConfirmingDataset'
    ]),
    ...mapActions([
      'createDataset',
      'createTestDataset',
      'confirmDataset',
      'confirmTestDataset'
    ]),
    nameInputNotify: function (e) {
      this.notifyNameField = (this.name.length === this.nameMaxLength)
      if (this.name.length > this.nameMaxLength) {
        this.name = this.name.substring(this.nameMaxLength)
      }
    },
    descriptionInputNotify: function (e) {
      this.notifyDescriptionField = (this.description.length === this.descriptionMaxLength)
    },
    onConfirmDataset: function () {
      const date = new Date()
      this.timeStamp = date.getTime()
      this.setConfirmingFlag(true)
      if (this.isTestDataset) {
        this.confirmTestDataset({
          'name': this.name,
          'ratio': this.ratio,
          'description': this.description,
        })
      } else {
        let test_dataset_id = this.test_dataset.id
        if (!test_dataset_id) {
          test_dataset_id = -1
        }
        this.confirmDataset({
          'hash': this.timeStamp,
          'name': this.name,
          'ratio': this.ratio,
          'description': this.description,
          'test_dataset_id': test_dataset_id,
        })
      }
    },
    onAddDataset: function () {
      if (this.isTestDataset) {
        this.createTestDataset({
          'name': this.name,
          'ratio': this.ratio,
          'description': this.description,
        }).then(() => {
          this.reset()
        })
      } else {
        let test_dataset_id = this.test_dataset.id
        if (!test_dataset_id) {
          test_dataset_id = -1
        }
        this.createDataset({
          'hash': this.timeStamp,
          'name': this.name,
          'ratio': this.ratio,
          'description': this.description,
          'test_dataset_id': test_dataset_id,
        }).then(() => {
          this.reset()
          this.$parent.showAddModel()
        })
      }
    },
    reset: function () {
      this.name = ''
      this.description = ''
      this.ratio = 0.8
      this.isTestDataset = false
      this.test_dataset = ''
      this.timeStamp = ''
      this.setConfirmingDataset(null)
    }
  }
}
</script>

<style lang='scss'>
#modal-add-dataset {
  display: flex;
  width: 100%;
  height: 100%;
  padding: 10px;
  font-size: $component-font-size-small;
  #title {
    width: 100%;
    height: 5%;
    color: gray;
    font-size: $component-font-size;
  }
  #dataset-setting {
    height: 100%;
    width: 50%;
    position: relative;
    .item {
      height: 10%;
      display: flex;
      align-items: center;
      justify-content: space-between;
      width: calc(100% - 8%);
      input, textarea {
        width: 50%;
      }
      select {
        background-color: white;
      }
      .warn {
        color: red;
        font-size: 0.4rem;
      }
    }
    input[type="button"] {
      width: 20%;
      margin-left: 74%;
      bottom: 0;
      position: absolute;
    }
  }
  #dataset-confirm {
    width: 50%;
    height: 100%;
    display: flex;
    flex-wrap: wrap;
    justify-content: flex-end;
    #dataset-ratio-bar {
      width: calc(100% - 30px);
      height: 20px;
      margin-bottom: 2%;
      display: flex;
      margin-left: 30px;
      section {
        height: 20px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
      }
    }
    #dataset-params {
      width: calc(100% - 3%);
      display: flex;
      margin-top: 3%;
      margin-left: 3%;
      span {
        width: calc(55%);
      }
    }
    #dataset-numbers {
      width: calc(100% - 3%);
      display: flex;
      margin-top: 3%;
      margin-left: 3%;
      margin-bottom: 3%;
      .num-item {
        width: 30%;
        display: flex;
        .num-title {
        }
        .num {
          margin-left: 3%;
        }
      }
    }
    #breakdown {
      width: 100%;
      margin-top: 2%;
      height: calc(100% - 5% - 3% - 1.6rem - 3% - 1.6rem - 6% - 20px - 2% - 40px - 2%);
      overflow: auto;
      #class-ratio-bars {
        height: 18px;
        width: 100%;
        display: flex;
        span:nth-child(1) {
          width: 20%;
          display: flex;
          justify-content: flex-end;
          margin-right: 5px;
        }
        .bar {
          height: 10px;
          display: flex;
        }
        section {
          height: 100%;
          display: flex;
          align-items: center;
          justify-content: center;
          color: white;
        }
      }
      #load-progress {
        width: 100%;
        height: 100%;
        display: flex;
        padding-bottom: 10%;
        align-items: center;
        justify-content: center;
        .lds-ripple {
          display: inline-block;
          position: relative;
          width: 64px;
          height: 64px;
          top: 3px;
          left: 3px;
        }
        .lds-ripple div {
          position: absolute;
          border: 4px solid #777;
          opacity: 1;
          border-radius: 50%;
          animation: lds-ripple 1s cubic-bezier(0, 0.2, 0.8, 1) infinite;
          animation-fill-mode: both;
        }
        .lds-ripple div:nth-child(2) {
          animation-delay: -0.5s;
          animation-fill-mode: both;
        }
        @keyframes lds-ripple {
          0% {
            top: 28px;
            left: 28px;
            width: 0;
            height: 0;
            opacity: 1;
          }
          100% {
            top: -1px;
            left: -1px;
            width: 58px;
            height: 58px;
            opacity: 0;
          }
        }
      }
    }
    #submit-button {
    }
  }
  .color-train {
    background-color: #0762AD;
  }
  .color-valid {
    background-color: #EF8200;
  }
  .train-anime {
    animation: growXTrain 0.8s linear;
    animation-fill-mode: both;
  }
  .valid-anime {
    animation: growXValid 0.8s linear;
    animation-fill-mode: both;
  }
  .bar-anime {
    animation: growX 0.8s;
    animation-fill-mode: both;
    animation-iteration-count: 1;
  }
  @keyframes growX {
    0% {
      transform: translateX(-50%) scaleX(0);
    }
    100% {
      transform: translateX(0) scaleX(1);
    }
  }
  .show-short-period {
    animation: notifyAnimation ease-in 3s;
  }
  @keyframes notifyAnimation {
   0% {
      outline-color: red;
    }
   80% {
      outline-color: red;
   }
  }
}
</style>
