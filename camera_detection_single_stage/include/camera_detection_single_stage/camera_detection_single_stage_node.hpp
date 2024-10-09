#ifndef __CAMERA_DETECTION_SINGLE_STAGE_HPP_
#define __CAMERA_DETECTION_SINGLE_STAGE_HPP_

#include "rclcpp/rclcpp.hpp"
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <image_transport/image_transport.hpp>
#if __has_include("cv_bridge/cv_bridge.hpp")
#include "cv_bridge/cv_bridge.hpp"
#else
#include "cv_bridge/cv_bridge.h"
#endif
#include <image_transport/image_transport.hpp>
#include "camera_detection_single_stage/monoflex.hpp"
#include <visualization_msgs/msg/marker_array.hpp>
#include "camera_detection_single_stage/postprocess.hpp"

namespace camera_detection_single_stage
{
class CameraDetectionSingleStageNode : public rclcpp::Node
{
public:
    CameraDetectionSingleStageNode(const rclcpp::NodeOptions& options);

    void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg);
    void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg);
    void publishNetResult(std::vector<monoflex::BBOX> const& bbox_res, cv::Mat const& img, double const& process_time, std_msgs::msg::Header const& header);
    void publishExperiemtalWork(std::vector<monoflex::BBOX> const& bbox_res, std_msgs::msg::Header const& header);

private:
    image_transport::Subscriber image_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr sub_camera_info_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_marker_, pub_post_marker_;
    image_transport::Publisher image_pub_;
    std::shared_ptr<monoflex::MonoFlex3D> detector_;
    std::shared_ptr<monoflex::PostprocessCameraDetection> postprocessor_;
    bool debug_;
    int current_marker_id_ = 0;
    int num_prev_obj_marker_id_ = 0;
    int post_current_marker_id_ = 0;
    int post_num_prev_obj_marker_id_ = 0;
    bool initialized_ = false;
    cv::Mat distortion_ = (cv::Mat_<float>(1, 5) << 0., 0., 0., 0. ,0.);
};
}  // namespace camera_detection_single_stage
#endif  // __CAMERA_DETECTION_SINGLE_STAGE_HPP_
