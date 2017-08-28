#ifndef OGN_COLOR_LOSS_PREP_LAYER_HPP_
#define OGN_COLOR_LOSS_PREP_LAYER_HPP_

#include "caffe/layers/ogn_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class OGNColorLossPrepLayer : public OGNLayer<Dtype> {
 public:
  explicit OGNColorLossPrepLayer(const LayerParameter& param)
      : OGNLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "OGNColorLossPrep"; }
  // virtual inline int ExactNumBottomBlobs() const { return 1; }
  // virtual inline int MinTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

};

}  // namespace caffe

#endif  //OGN_COLOR_LOSS_PREP_LAYER_HPP_
