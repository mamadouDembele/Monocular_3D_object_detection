#include "camera_detection_single_stage/monoflex.hpp"
#include <regex>

namespace monoflex{
    
MonoFlex3D::MonoFlex3D(NETPARAMS const& params) : model_params_(params)
{
    /******* Create ORT environment *******/
    env_ = Ort::Env(ORT_LOGGING_LEVEL_INFO, "MonoFlex");
    
    /**************** Create ORT session ******************/
    Ort::SessionOptions sessionOption; // Set up options for session

    if (params.ep == EP::CUDA){
        OrtCUDAProviderOptions cudaOption;
        cudaOption.device_id = 0;
        sessionOption.AppendExecutionProvider_CUDA(cudaOption);
    }else if(params.ep == EP::TENSORRT){
        const auto& api = Ort::GetApi();
        OrtTensorRTProviderOptionsV2* tensorrt_options;
        Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&tensorrt_options));

        if(params.precision == PRECISION::FP16){
            std::vector<const char*> option_keys = {"device_id", "trt_fp16_enable", "trt_engine_cache_enable", "trt_engine_cache_path", "trt_context_memory_sharing_enable",};
            std::vector<const char*> option_values = {"0", "1", "1", "cache", "1",};
            Ort::ThrowOnError(api.UpdateTensorRTProviderOptions(tensorrt_options, option_keys.data(), option_values.data(), option_keys.size()));
        }else{
            std::vector<const char*> option_keys = {"device_id", "trt_fp16_enable", "trt_engine_cache_enable", "trt_engine_cache_path", "trt_context_memory_sharing_enable",};
            std::vector<const char*> option_values = {"0", "0", "1", "cache", "1",};
            Ort::ThrowOnError(api.UpdateTensorRTProviderOptions(tensorrt_options, option_keys.data(), option_values.data(), option_keys.size()));
        }
        cudaStream_t cuda_stream;
        cudaStreamCreate(&cuda_stream);
        const char* key_cuda_stream = "user_compute_stream";
        Ort::ThrowOnError(api.UpdateTensorRTProviderOptionsWithValue(tensorrt_options, key_cuda_stream, cuda_stream));
        sessionOption.AppendExecutionProvider_TensorRT_V2(*tensorrt_options);
    }

    sessionOption.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL); // Sets graph optimization level (Here, enable all possible optimizations)
    sessionOption.SetIntraOpNumThreads(params.intraOpNumThreads);
    sessionOption.SetLogSeverityLevel(params.logSeverityLevel);
    const char* modelPath = params.model_path.c_str();
    session_ = new Ort::Session(env_, modelPath, sessionOption); // Create session by loading the onnx model

    /**************** Create allocator ******************/
    Ort::AllocatorWithDefaultOptions allocator; // Allocator is used to get model information
    std::cout << "******* [MonoFlex3D]: Model information below *******" << std::endl;

    /**************** Input info ******************/
    size_t inputNodesNum = session_->GetInputCount();  // Get the number of input nodes  
    std::cout << "[MonoFlex3D]: Number Input Nodes: " << inputNodesNum << std::endl;

    for (size_t i = 0; i < inputNodesNum; i++)
    {
        // Get the name of the inputs
        Ort::AllocatedStringPtr input_node_name = session_->GetInputNameAllocated(i, allocator);
        char* temp_buf = new char[50];
        strcpy(temp_buf, input_node_name.get());
        input_node_names_.push_back(temp_buf);
        std::cout << "[MonoFlex3D]: Input " << i << " Name: " << temp_buf << std::endl;

        // Get the type of the input
        Ort::TypeInfo inputTypeInfo = session_->GetInputTypeInfo(i);
        auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
        ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();
        std::cout << "[MonoFlex3D]: Input " << i << " Type: " << inputType << std::endl;

        // Get the shape of the input
        auto mInputDims = inputTensorInfo.GetShape();
        input_shapes_.push_back(mInputDims);
        std::cout << "[MonoFlex3D]: Input " << i << " Dimensions: " << mInputDims << std::endl;
    }

    /**************** Output info ******************/    
    size_t OutputNodesNum = session_->GetOutputCount(); // Get the number of outputs nodes 
    std::cout << "[MonoFlex3D]: Number Outputs Nodes: " << OutputNodesNum << std::endl;

    for (size_t i = 0; i < OutputNodesNum; i++)
    {
        // Get the name of the outputs
        Ort::AllocatedStringPtr output_node_name = session_->GetOutputNameAllocated(i, allocator);
        char* temp_buf = new char[10];
        strcpy(temp_buf, output_node_name.get());
        output_node_names_.push_back(temp_buf);
        std::cout << "[MonoFlex3D]: Output " << i << " Name: " << temp_buf << std::endl;

        // Get the type of the output
        Ort::TypeInfo outputTypeInfo = session_->GetOutputTypeInfo(i);
        auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
        ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();
        std::cout << "[MonoFlex3D]: Output " << i << " Type: " << outputType << std::endl;

        // Get the shape of the output
        auto mOutputDims = outputTensorInfo.GetShape();
        output_shapes_.push_back(mOutputDims);
        std::cout << "[MonoFlex3D]: Output " << i << " Dimensions: " << mOutputDims << std::endl;
    }
    options_ = Ort::RunOptions{ nullptr };
    camera_intrinsec_ = (cv::Mat_<float>(3, 3) << 721.54, 0.0, 609.56, 0.0, 721.54, 172.8, 0.0, 0.0, 1.0);
    WarmUpSession();
}


MonoFlex3D::~MonoFlex3D() {
    delete session_;
}

void MonoFlex3D::FilterByMinDims(std::vector<BBOX>& bbox_vec){
    std::size_t valid_index = 0, cur_index = 0;
    while (cur_index < bbox_vec.size()) {
        const auto& bb = bbox_vec.at(cur_index);
        float height = bb.box.height;
        if (height >= model_params_.min_2d_height && bb.whl.x >= model_params_.min_3d_width && bb.whl.y >= model_params_.min_3d_height && bb.whl.z >= model_params_.min_3d_length) {
            if (valid_index != cur_index)
                bbox_vec.at(valid_index) = bbox_vec.at(cur_index);
            ++valid_index;
        }
        ++cur_index;
    }
    bbox_vec.resize(valid_index);
}

template<typename N>
void MonoFlex3D::TensorProcess(N const& blob_img, N const& blob_k, size_t img_size, size_t k_size, std::vector<BBOX>& bbox_res) {

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(Ort::Value::CreateTensor<typename std::remove_pointer<N>::type>(memoryInfo, blob_img, img_size, input_shapes_[0].data(), input_shapes_[0].size()));
    input_tensors.push_back(Ort::Value::CreateTensor<typename std::remove_pointer<N>::type>(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), blob_k, k_size, input_shapes_[1].data(), input_shapes_[1].size()));

    auto output_tensors = session_->Run(options_, input_node_names_.data(), input_tensors.data(), input_node_names_.size(), output_node_names_.data(), output_node_names_.size());

    Ort::TypeInfo typeInfo = output_tensors[0].GetTypeInfo();
    auto tensor_info = typeInfo.GetTensorTypeAndShapeInfo();
    int number_pred = tensor_info.GetShape()[0];

    // Get predictions
    auto scores = output_tensors[0].GetTensorMutableData<typename std::remove_pointer<N>::type>();
    auto bboxes = output_tensors[1].GetTensorMutableData<typename std::remove_pointer<N>::type>();
    auto cls_indexes = output_tensors[2].GetTensorMutableData<typename std::remove_pointer<int64_t>::type>();

    for (int i(0); i < number_pred; i++){
        if (*(scores + i) < model_params_.confidence_threshold){
            continue;
        }
        auto bbox = bboxes + i * 12;
        BBOX b;
        b.score = *(scores + i);
        b.label = static_cast<LABELS>(*(cls_indexes + i));
        b.whl = cv::Point3f(*(bbox + 7), *(bbox + 8), *(bbox + 9));
        b.xyz = cv::Point3f(*(bbox + 4), *(bbox + 5), *(bbox + 6));
        b.theta = *(bbox + 11);
        b.alpha = *(bbox + 10);
        float x1 = *(bbox + 0) / scale_;
        float y1 = *(bbox + 1) / scale_;
        float x2 = *(bbox + 2) / scale_;
        float y2 = *(bbox + 3) / scale_;

        int left = int(x1);
        int top = int(y1);

        int width = int(abs(x1 - x2));
        int height = int(abs(y1 - y2));
        b.box = cv::Rect(left, top, width, height);
        bbox_res.push_back(b);
    }
}

void MonoFlex3D::RunSession(cv::Mat const& input_img, std::vector<BBOX>& bbox_res){

    // Get the inputs size
    std::vector<int64_t> inputImgNodeDims = input_shapes_[0]; // img shape
    std::vector<int64_t> inputKNodeDims = input_shapes_[1]; // k shape

    // Compute the product of all input dimension
    size_t img_inputTensorSize = vectorProduct(inputImgNodeDims);
    size_t k_inputTensorSize = vectorProduct(inputKNodeDims);

    if (model_params_.ep == EP::CUDA && model_params_.precision == PRECISION::FP16){
        half* blob_img = new half[img_inputTensorSize];
        half* blob_k = new half[k_inputTensorSize];
        PreProcess(input_img, blob_img, blob_k);
        TensorProcess(blob_img, blob_k, img_inputTensorSize, k_inputTensorSize, bbox_res);

        delete[] blob_img;
        delete[] blob_k;
    }else{
        float* blob_img = new float[img_inputTensorSize];
        float* blob_k = new float[k_inputTensorSize];
        PreProcess(input_img, blob_img, blob_k);
        TensorProcess(blob_img, blob_k, img_inputTensorSize, k_inputTensorSize, bbox_res);

        delete[] blob_img;
        delete[] blob_k;
    }
}

template<typename T>
void MonoFlex3D::PreProcess(cv::Mat const& input_img, T& out_img_tensor, T& out_k)
{
    // Get the inputs size of the network
    std::vector<int64_t> net_image_shape = input_shapes_[0]; // img shape
    std::vector<int64_t> net_K_shape = input_shapes_[1]; // k shape

    // Compute the relative scale_ between image and network
    double height = static_cast<double>(input_img.rows);
    double width = static_cast<double>(input_img.cols);
    scale_ = std::min(static_cast<double>(net_image_shape[2])/height, static_cast<double>(net_image_shape[3])/width);
    int height_eff = static_cast<int>(height * scale_);
    int width_eff = static_cast<int>(width * scale_);

    // resize input image base on the scale_
    cv::Mat tmp_img;
    cv::resize(input_img, tmp_img, cv::Size(width_eff, height_eff));

    // make border to reach net shape
    int down = (net_image_shape.at(2) - height_eff);
    int right = (net_image_shape.at(3) - width_eff);
    cv::copyMakeBorder(tmp_img, tmp_img, 0, down, 0, right, cv::BORDER_CONSTANT, {0, 0, 0});

    // preprocess
    tmp_img.convertTo(tmp_img, CV_32F, 1.0f / 255.0);
    std::vector<float> mean_values{0.485, 0.456, 0.406};
    std::vector<float> std_values{0.229, 0.224, 0.225};

    std::vector<cv::Mat> bgrChannels(3);
    cv::split(tmp_img, bgrChannels);
    for (int i = 0; i < 3; ++i){
        bgrChannels[i].convertTo(bgrChannels[i], CV_32FC1, 1 / std_values[i], (0.0 - mean_values[i]) / std_values[i]);
    }

    cv::Mat out_img;
    cv::merge(bgrChannels, out_img);

    // Convert HWC to CHW
    cv::dnn::blobFromImage(out_img, out_img);

    // fill image tensor
    std::vector<float> vector;
    vector.assign(out_img.begin<float>(), out_img.end<float>());
    std::copy(vector.begin(), vector.end(), out_img_tensor);

    // fill camera intrinsec tensor
    out_k[0] = camera_intrinsec_.at<float>(0, 0) * scale_;
    out_k[1] = camera_intrinsec_.at<float>(0, 1) * scale_;
    out_k[2] = camera_intrinsec_.at<float>(0, 2) * scale_;
    out_k[3] = 0.;
    out_k[4] = camera_intrinsec_.at<float>(1, 0) * scale_;
    out_k[5] = camera_intrinsec_.at<float>(1, 1) * scale_;
    out_k[6] = camera_intrinsec_.at<float>(1, 2) * scale_;
    out_k[7] = 0.;
    out_k[8] = camera_intrinsec_.at<float>(2, 0);
    out_k[9] = camera_intrinsec_.at<float>(2, 1);
    out_k[10] = camera_intrinsec_.at<float>(2, 2);
    out_k[11] = 0.0;
}

void MonoFlex3D::WarmUpSession() {
    clock_t starttime_1 = clock();

    // Get the inputs size
    std::vector<int64_t> inputImgNodeDims = input_shapes_[0]; // img shape
    std::vector<int64_t> inputKNodeDims = input_shapes_[1]; // k shape

    // Compute the product of all input dimension
    size_t img_inputTensorSize = vectorProduct(inputImgNodeDims);
    size_t k_inputTensorSize = vectorProduct(inputKNodeDims);
    cv::Mat input_img = cv::imread(model_params_.sample_path);
    std::vector<Ort::Value> input_tensors;
    
    if (model_params_.ep == EP::CUDA && model_params_.precision == PRECISION::FP16){

        half* blob_img = new half[img_inputTensorSize];
        half* blob_k = new half[k_inputTensorSize];
        PreProcess(input_img, blob_img, blob_k);
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        input_tensors.push_back(Ort::Value::CreateTensor<half>(memoryInfo, blob_img, img_inputTensorSize, inputImgNodeDims.data(), inputImgNodeDims.size()));
        input_tensors.push_back(Ort::Value::CreateTensor<half>(memoryInfo, blob_k, k_inputTensorSize, inputKNodeDims.data(), inputKNodeDims.size()));

        auto output_tensors = session_->Run(options_, input_node_names_.data(), input_tensors.data(), input_node_names_.size(), output_node_names_.data(), output_node_names_.size());

        delete[] blob_img;
        delete[] blob_k;
        clock_t starttime_4 = clock();
        double post_process_time = (double)(starttime_4 - starttime_1) / CLOCKS_PER_SEC * 1000;
        std::cout << "[MONOFLEX]: " << "Cuda warm-up cost " << post_process_time << " ms. " << std::endl;
    }else{
        float* blob_img = new float[img_inputTensorSize];
        float* blob_k = new float[k_inputTensorSize];
        PreProcess(input_img, blob_img, blob_k);
        
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        input_tensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, blob_img, img_inputTensorSize, inputImgNodeDims.data(), inputImgNodeDims.size()));
        input_tensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, blob_k, k_inputTensorSize, inputKNodeDims.data(), inputKNodeDims.size()));
        auto output_tensors = session_->Run(options_, input_node_names_.data(), input_tensors.data(), input_node_names_.size(), output_node_names_.data(), output_node_names_.size());

        delete[] blob_img;
        delete[] blob_k;
        clock_t starttime_4 = clock();
        double post_process_time = (double)(starttime_4 - starttime_1) / CLOCKS_PER_SEC * 1000;
        std::cout << "[MONOFLEX]: " << "Cuda warm-up cost " << post_process_time << " ms. " << std::endl;
    }
}
}