#pragma once
#include "object.hpp"

namespace monoflex{

class MonoFlex3D
{
public:

    /**
     * @brief Constructor
     * @param params: model param
   */
    MonoFlex3D(NETPARAMS const& params);

    ~MonoFlex3D();

    void RunSession(cv::Mat const& img, std::vector<BBOX>& bbox_res);
    void WarmUpSession();

    template<typename N>
    void TensorProcess(N const& blob_img, N const& blob_k, size_t img_size, size_t k_size, std::vector<BBOX>& bbox_res);

    template<typename T>
    void PreProcess(cv::Mat const& input_img, T& out_img_tensor, T& out_k);

    void setupCameraIntrinsec(cv::Mat const& k) {camera_intrinsec_ = k;}
    cv::Mat getCameraIntrinsec(){return camera_intrinsec_;}

    void FilterByMinDims(std::vector<BBOX>& bbox_vec);

private:

    // ORT Environment
    Ort::Env env_;

    // Session
    Ort::Session* session_;

    // run options
    Ort::RunOptions options_;

    // inputs
    std::vector<const char*> input_node_names_;
    std::vector<std::vector<int64_t>> input_shapes_;
    
    // outputs
    std::vector<const char*> output_node_names_;
    std::vector<std::vector<int64_t>> output_shapes_;

    // constmodel params
    NETPARAMS model_params_;

    // intrinsec
    cv::Mat camera_intrinsec_;

    // ratio between net input and 
    double scale_;
};
}
