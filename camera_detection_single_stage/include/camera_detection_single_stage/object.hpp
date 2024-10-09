#pragma once
#include <string>
#include <vector>
#include <cstdio>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include "onnxruntime_cxx_api.h"
#include <cuda_fp16.h>
#include "cuda_runtime_api.h"
#include <array>
#include <memory>
#include <numeric>
#include <map>
#include <filesystem>

namespace Ort
{
    template<>
    struct TypeToTensorType<half> { static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16; };
}


namespace monoflex{

// Execution provider (CPU, CUDA, or TENSORRT)
enum class EP
{
    CPU = 1,
    CUDA = 2,
    TENSORRT = 3,
};

// if EP == CUDA or TENSORRT
enum class PRECISION
{
    FP32 = 1,
    FP16 = 2,
};

enum class LABELS
{
    CAR = 0, 
    TRUCK = 1, 
    BUS = 2, 
    TRAILER = 3, 
    CONSTRUCTION_VEHICLE = 4,
    PEDESTRIAN = 5, 
    MOTORCYCLE = 6, 
    BICYCLE = 7,
    TRAFFIC_CONE = 8, 
    BARRIER = 9,
};

struct NETPARAMS
{
    std::string model_path; // path of the model
    std::string sample_path;
    EP ep;
    PRECISION precision; // precision
    float confidence_threshold; // threshold
    float min_2d_height;
    float min_3d_height;
    float min_3d_width;
    float min_3d_length;
    int logSeverityLevel = 3;
    int intraOpNumThreads = 8;
};

// output of the network
struct BBOX
{
    LABELS label;
    float score;
    cv::Rect box;
    cv::Point3f whl;
    cv::Point3f xyz;
    float theta;
    float alpha;
};

const std::map<LABELS, std::string> mapLabelString = {
    {LABELS::CAR, "CAR: "},
    {LABELS::CONSTRUCTION_VEHICLE, "CONSTRUCTION_VEHICLE: "},
    {LABELS::BUS, "BUS: "},
    {LABELS::TRUCK, "TRUCK: "},
    {LABELS::BICYCLE, "BICYCLE: "},
    {LABELS::MOTORCYCLE, "MOTORCYCLE: "},
    {LABELS::PEDESTRIAN, "PEDESTRIAN: "},
    {LABELS::TRAFFIC_CONE, "TRAFFICCONE: "},
    {LABELS::BARRIER, "BARRIER: "},
    {LABELS::TRAILER, "TRAILER: "},
};

/**
 * @brief Operator overloading for printing vectors
 * @tparam T
 * @param os
 * @param v
 * @return std::ostream&
 */
template <typename T>
static std::ostream& operator<<(std::ostream& os, const std::vector<T>& v) {
  os << "[";
  for (size_t i = 0; i < v.size(); ++i) {
    os << v[i];
    if (i != v.size() - 1) {
      os << ", ";
    }
  }
  os << "]";
  return os;
}


/**
 * @brief Compute the product over all the elements of a vector
 * @tparam T
 * @param v: input vector
 * @return the product
 */
template <typename T>
static size_t vectorProduct(const std::vector<T>& v) {
  return std::accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}
}