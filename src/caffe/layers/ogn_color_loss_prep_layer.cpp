#include "caffe/layers/ogn_color_loss_prep_layer.hpp"
#include "caffe/layers/ogn_color_data_layer.hpp"
#include "caffe/net.hpp"

#include "image_tree_tools/image_tree_tools.h"

namespace caffe {

using namespace std;

template <typename Dtype>
void OGNLossPrepLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void OGNLossPrepLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    int batch_size = bottom[0]->shape(0);
    int num_pixels = bottom[0]->shape(2);

    vector<int> output_shape;
    output_shape.push_back(batch_size);
    output_shape.push_back(3);  // Since 3 color channels for the colored output case
    output_shape.push_back(num_pixels);
    output_shape.push_back(1);

    top[0]->Reshape(output_shape);
    if(top.size() == 2) top[1]->Reshape(output_shape);

    if(top.size() < 1 || top.size() > 2) LOG(FATAL) << "Wrong number of output blobs.";
}

template <typename Dtype>
void OGNLossPrepLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    int batch_size = bottom[0]->shape(0);
    int num_pixels = bottom[0]->shape(2);

    int gt_num_pixels = bottom[1]->shape(1);

    const string gt_key_layer_name = this->layer_param_.ogn_loss_prep_param().gt_key_layer();
    const string pr_key_layer_name = this->layer_param_.ogn_loss_prep_param().pr_key_layer();
    const bool use_voxel_grid = this->layer_param_.ogn_loss_prep_param().use_voxel_grid();

    shared_ptr<Layer<Dtype> > gt_raw_ptr = this->parent_net()->layer_by_name(gt_key_layer_name);
    shared_ptr<Layer<Dtype> > pr_raw_ptr = this->parent_net()->layer_by_name(pr_key_layer_name);

    shared_ptr<OGNLayer<Dtype> > gt_key_layer = boost::dynamic_pointer_cast<OGNLayer<Dtype> >(gt_raw_ptr);
    shared_ptr<OGNLayer<Dtype> > pr_key_layer = boost::dynamic_pointer_cast<OGNLayer<Dtype> >(pr_raw_ptr);

    const Dtype* gt_values = bottom[1]->cpu_data();

    Dtype* output_colors = top[0]->mutable_cpu_data();

    const int dim = batch_size * top[0]->shape(1) * 3 * top[0]->shape(2);
    //caffe_set(dim, (Dtype)CLASS_IGNORE, output_classification);
    caffe_set( dim , (Dtype)0, output_colors )

    for(int bt = 0; bt<batch_size; bt++)
    {
        //multi-class classification
        if(top.size() == 1)
        {
            GeneralOctree<int> &pr_keys_octree = pr_key_layer->get_keys_octree(bt);
            GeneralOctree<int> &pr_prop_octree = pr_key_layer->get_prop_octree(bt);
            GeneralOctree<int> &gt_keys_octree = gt_key_layer->get_keys_octree(bt);

            for(GeneralOctree<int>::iterator it=pr_keys_octree.begin(); it!=pr_keys_octree.end(); it++)
            {
                if(pr_prop_octree.get_value(it->first) == PROP_TRUE)
                {
                    SignalType gt_value;
                    int gt_ind = gt_keys_octree.get_value(it->first, use_voxel_grid);
                    for( ch=0;ch<3;ch++){
                        if(gt_ind != -1) gt_value = gt_values[bt *3*gt_num_pixels + ch*gt_num_pixels + gt_ind];
                        else gt_value = 0;           // CLASS_MIXED;
                        output_colors[bt * 3 * num_pixels + ch*num_pixels + it->second  ] = gt_value;
                    }
                }
                else
                {
                    output_colors[bt * num_pixels + ch*num_pixels  + it->second] = 0;            // CLASS_IGNORE;   
                }
            }
        }
        //regression
        else if(top.size() == 2)
        {
        }
    }
}

template <typename Dtype>
void OGNLossPrepLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}

INSTANTIATE_CLASS(OGNColorLossPrepLayer);
REGISTER_LAYER_CLASS(OGNColorLossPrep);

}  // namespace caffe
