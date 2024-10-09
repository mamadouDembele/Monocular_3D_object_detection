#include "camera_detection_single_stage/camera_detection_single_stage_node.hpp"

static float min_process_time(1000000000.0), max_process_time(0);

namespace camera_detection_single_stage
{
CameraDetectionSingleStageNode::CameraDetectionSingleStageNode(const rclcpp::NodeOptions& options) : 
Node("camera_detection_single_stage_node", options)
{
    // Network params
    monoflex::NETPARAMS params_net;
    params_net.model_path = this->declare_parameter<std::string>("model_path");
    params_net.sample_path = this->declare_parameter<std::string>("sample_path");
    params_net.confidence_threshold = this->declare_parameter<double>("confidence_threshold");
    params_net.min_2d_height = this->declare_parameter<double>("min_2d_height");
    params_net.min_3d_height = this->declare_parameter<double>("min_3d_height");
    params_net.min_3d_width = this->declare_parameter<double>("min_3d_width");
    params_net.min_3d_length = this->declare_parameter<double>("min_3d_length");
    params_net.ep = monoflex::EP::TENSORRT;
    params_net.precision = monoflex::PRECISION::FP16;

    // postprocess params
    monoflex::TransformerParams params_post;
    params_post.max_nr_iter = this->declare_parameter<int>("max_nr_iter");
    params_post.learning_rate = this->declare_parameter<float>("learning_rate");
    params_post.k_min_cost = this->declare_parameter<float>("k_min_cost");
    params_post.eps_cost = this->declare_parameter<float>("eps_cost");

    debug_ = this->declare_parameter<bool>("debug");
    
    detector_ = std::make_shared<monoflex::MonoFlex3D>(params_net);
    postprocessor_ = std::make_shared<monoflex::PostprocessCameraDetection>(params_post);

    using std::placeholders::_1;
    image_sub_ = image_transport::create_subscription(this, "input/image", std::bind(&CameraDetectionSingleStageNode::imageCallback, this, _1), "raw", rmw_qos_profile_sensor_data);
    sub_camera_info_ = this->create_subscription<sensor_msgs::msg::CameraInfo>("input/camera_info", 1, std::bind(&CameraDetectionSingleStageNode::cameraInfoCallback, this, _1));
    pub_marker_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("~/debug/marker", 1);
    pub_post_marker_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("~/debug/exp_marker", 1);
    if(debug_){
        image_pub_ = image_transport::create_publisher(this, "~/debug/image", rclcpp::QoS{1}.get_rmw_qos_profile());
    }
}

void CameraDetectionSingleStageNode::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg){

    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception &e) {
        RCLCPP_ERROR(this->get_logger(), "Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
        return;
    }

    // detection
    auto start = std::chrono::system_clock::now();
    cv::Mat img = cv_ptr->image;
    std::vector<monoflex::BBOX> bbox_res;
    detector_->RunSession(img, bbox_res);
    detector_->FilterByMinDims(bbox_res);
    auto end = std::chrono::system_clock::now();
    double process_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

    // visualization
    publishNetResult(bbox_res, img, process_time, msg->header);

    // experiemtal
    postprocessor_->transform(bbox_res, detector_->getCameraIntrinsec(), distortion_);

    publishExperiemtalWork(bbox_res, msg->header);

}

void CameraDetectionSingleStageNode::cameraInfoCallback(const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg){
    cv::Mat intrinsic = (cv::Mat_<float>(3, 3) << msg->k[0], msg->k[1], msg->k[2],msg->k[3], msg->k[4], msg->k[5], msg->k[6], msg->k[7], msg->k[8]);
    detector_->setupCameraIntrinsec(intrinsic);
    for(int i(0); i < 5; i++){
        distortion_.at<float>(0, i) = msg->d[i];
    }
}

void CameraDetectionSingleStageNode::publishNetResult(std::vector<monoflex::BBOX> const& bbox_res, cv::Mat const& img, 
                                                                double const& process_time, std_msgs::msg::Header const& header){
    cv::Mat K = detector_->getCameraIntrinsec();;
    cv::Mat out = img.clone();
    
    visualization_msgs::msg::MarkerArray marker_array;
    int marker_id = 0;
    for (auto const& b: bbox_res){
        if(debug_){
            cv::Scalar color = cv::Scalar(0, 0, 142);;
            std::string text = monoflex::mapLabelString.at(b.label) + std::to_string(b.score);           
            cv::rectangle(out, b.box, color, 2);
            cv::putText(out, text, cv::Point2d(b.box.x, b.box.y -10), cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv::LINE_AA);
        }

        float w = b.whl.x;
        float h = b.whl.y;
        float l = b.whl.z;
        float angle = b.theta;
        float t[3] = {b.xyz.z, b.xyz.y, b.xyz.x};

        float rot_y[9] = {0};
        float x_corners[8] = {0};
        float y_corners[8] = {0};
        float z_corners[8] = {0};
        monoflex::GenCorners(h, w, l, x_corners, y_corners, z_corners);
        monoflex::GenRotMatrix(angle, rot_y);

        visualization_msgs::msg::Marker marker;
        marker.header = header;
        marker.id = marker_id;
        marker.type = visualization_msgs::msg::Marker::LINE_LIST;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.scale.x = 0.2;
        marker.color.a = 1.0;
        marker.color.r = 0;
        marker.color.g = 0;
        marker.color.b = 1.0;
        marker.lifetime = rclcpp::Duration(0, 1e-8);

        cv::Mat corners = cv::Mat_<float>(8, 3);

        // rotation of the eight
        for (int i = 0; i < 8; ++i) {
            // Bbox from x_images
            float x_box[3] = {x_corners[i], y_corners[i], z_corners[i]};
            float res[3];
            monoflex::IProjectThroughExtrinsic(rot_y, t, x_box, res);
            corners.at<float>(i, 0) = res[2];
            corners.at<float>(i, 1) = res[1];
            corners.at<float>(i, 2) = res[0];
        }
        
        geometry_msgs::msg::Point pt;
        for (int i = 1; i < 5; ++i) {
            pt.x = corners.at<float>(i, 0);
            pt.y = corners.at<float>(i, 1);
            pt.z = corners.at<float>(i, 2);
            marker.points.push_back(pt);
            pt.x = corners.at<float>(i%4+1, 0);
            pt.y = corners.at<float>(i%4+1, 1);
            pt.z = corners.at<float>(i%4+1, 2);
            marker.points.push_back(pt);
            pt.x = corners.at<float>((i+4)%8, 0);
            pt.y = corners.at<float>((i+4)%8, 1);
            pt.z = corners.at<float>((i+4)%8, 2);
            marker.points.push_back(pt);
            pt.x = corners.at<float>(((i)%4 + 5)%8, 0);
            pt.y = corners.at<float>(((i)%4 + 5)%8, 1);
            pt.z = corners.at<float>(((i)%4 + 5)%8, 2);
            marker.points.push_back(pt);
        }

        pt.x = corners.at<float>(2, 0);
        pt.y = corners.at<float>(2, 1);
        pt.z = corners.at<float>(2, 2);
        marker.points.push_back(pt);
        pt.x = corners.at<float>(7, 0);
        pt.y = corners.at<float>(7, 1);
        pt.z = corners.at<float>(7, 2);
        marker.points.push_back(pt);
        pt.x = corners.at<float>(3, 0);
        pt.y = corners.at<float>(3, 1);
        pt.z = corners.at<float>(3, 2);
        marker.points.push_back(pt);
        pt.x = corners.at<float>(6, 0);
        pt.y = corners.at<float>(6, 1);
        pt.z = corners.at<float>(6, 2);
        marker.points.push_back(pt);
        pt.x = corners.at<float>(4, 0);
        pt.y = corners.at<float>(4, 1);
        pt.z = corners.at<float>(4, 2);
        marker.points.push_back(pt);
        pt.x = corners.at<float>(5, 0);
        pt.y = corners.at<float>(5, 1);
        pt.z = corners.at<float>(5, 2);
        marker.points.push_back(pt);
        pt.x = corners.at<float>(0, 0);
        pt.y = corners.at<float>(0, 1);
        pt.z = corners.at<float>(0, 2);
        marker.points.push_back(pt);
        pt.x = corners.at<float>(1, 0);
        pt.y = corners.at<float>(1, 1);
        pt.z = corners.at<float>(1, 2);
        marker.points.push_back(pt);

        marker_array.markers.push_back(marker);
        marker_id++;

        if(debug_){
            std::vector<cv::Point2f> img_corners(8);
            for (int i = 0; i < 8; ++i) {
                float x = K.at<float>(0, 0) * corners.at<float>(i, 0) + K.at<float>(0, 1) *  corners.at<float>(i, 1) + K.at<float>(0, 2) *  corners.at<float>(i, 2);
                float y = K.at<float>(1, 1) * corners.at<float>(i, 1) + K.at<float>(1, 2) *  corners.at<float>(i, 2);
                img_corners[i].x = x / corners.at<float>(i, 2);
                img_corners[i].y = y / corners.at<float>(i, 2);
            }

            for (int i = 1; i < 5; ++i) {
                const auto& p1 = img_corners[i];
                const auto& p2 = img_corners[i%4 + 1];
                const auto& p3 = img_corners[(i+4)%8];
                const auto& p4 = img_corners[((i)%4 + 5)%8];
                cv::line(out, p1, p2, cv::Scalar(241, 101, 72), 2, cv::LINE_AA);
                cv::line(out, p3, p4, cv::Scalar(241, 101, 72), 2, cv::LINE_AA);
            }
            cv::line(out, img_corners[2], img_corners[7], cv::Scalar(241, 101, 72), 2, cv::LINE_AA);
            cv::line(out, img_corners[3], img_corners[6], cv::Scalar(241, 101, 72), 2, cv::LINE_AA);
            cv::line(out, img_corners[4], img_corners[5], cv::Scalar(241, 101, 72), 2, cv::LINE_AA);
            cv::line(out, img_corners[0], img_corners[1], cv::Scalar(241, 101, 72), 2, cv::LINE_AA);
        }
    }

    if(debug_){
        std::ostringstream strs;
        strs << process_time;
        std::string text = "inference time: " + strs.str();
        cv::putText(out, text, cv::Point2d(100, 100), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
        if (process_time < min_process_time){
            min_process_time = process_time;
        }
        else if (process_time > max_process_time){
            max_process_time = process_time;
        }
        RCLCPP_INFO_STREAM(this->get_logger(), "min process time: " << min_process_time << "ms" 
                                                << " max process time: " << max_process_time << "ms"
                                                << " current process time: " << process_time << "ms");
        sensor_msgs::msg::Image::SharedPtr image_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", out).toImageMsg();
        image_pub_.publish(image_msg);
    }

    current_marker_id_ = marker_array.markers.size();
    for (int id_num_diff = num_prev_obj_marker_id_-current_marker_id_; id_num_diff > 0; id_num_diff--)
    {
        visualization_msgs::msg::Marker m;
        m.id = id_num_diff + current_marker_id_ - 1;
        m.action = m.DELETE;
        marker_array.markers.push_back(m);
    }
    num_prev_obj_marker_id_ = current_marker_id_; // For the next callback */
    pub_marker_->publish(marker_array);
}

void CameraDetectionSingleStageNode::publishExperiemtalWork(std::vector<monoflex::BBOX> const& bbox_res, std_msgs::msg::Header const& header){
    cv::Mat K = detector_->getCameraIntrinsec();;
    visualization_msgs::msg::MarkerArray marker_array;
    int marker_id = 0;
    for (auto const& b: bbox_res){
        float w = b.whl.x;
        float h = b.whl.y;
        float l = b.whl.z;
        float angle = b.theta;
        float t[3] = {b.xyz.z, b.xyz.y, b.xyz.x};

        float rot_y[9] = {0};
        float x_corners[8] = {0};
        float y_corners[8] = {0};
        float z_corners[8] = {0};
        monoflex::GenCorners(h, w, l, x_corners, y_corners, z_corners);
        monoflex::GenRotMatrix(angle, rot_y);

        visualization_msgs::msg::Marker marker;
        marker.header = header;
        marker.id = marker_id;
        marker.type = visualization_msgs::msg::Marker::LINE_LIST;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.scale.x = 0.2;
        marker.color.a = 1.0;
        marker.color.r = 1.0;
        marker.color.g = 0;
        marker.color.b = 1.0;
        marker.lifetime = rclcpp::Duration(0, 1e-8);

        cv::Mat corners = cv::Mat_<float>(8, 3);

        // rotation of the eight
        for (int i = 0; i < 8; ++i) {
            // Bbox from x_images
            float x_box[3] = {x_corners[i], y_corners[i], z_corners[i]};
            float res[3];
            monoflex::IProjectThroughExtrinsic(rot_y, t, x_box, res);
            corners.at<float>(i, 0) = res[2];
            corners.at<float>(i, 1) = res[1];
            corners.at<float>(i, 2) = res[0];
        }
        

        geometry_msgs::msg::Point pt;
        for (int i = 1; i < 5; ++i) {
            pt.x = corners.at<float>(i, 0);
            pt.y = corners.at<float>(i, 1);
            pt.z = corners.at<float>(i, 2);
            marker.points.push_back(pt);
            pt.x = corners.at<float>(i%4+1, 0);
            pt.y = corners.at<float>(i%4+1, 1);
            pt.z = corners.at<float>(i%4+1, 2);
            marker.points.push_back(pt);
            pt.x = corners.at<float>((i+4)%8, 0);
            pt.y = corners.at<float>((i+4)%8, 1);
            pt.z = corners.at<float>((i+4)%8, 2);
            marker.points.push_back(pt);
            pt.x = corners.at<float>(((i)%4 + 5)%8, 0);
            pt.y = corners.at<float>(((i)%4 + 5)%8, 1);
            pt.z = corners.at<float>(((i)%4 + 5)%8, 2);
            marker.points.push_back(pt);
        }

        pt.x = corners.at<float>(2, 0);
        pt.y = corners.at<float>(2, 1);
        pt.z = corners.at<float>(2, 2);
        marker.points.push_back(pt);
        pt.x = corners.at<float>(7, 0);
        pt.y = corners.at<float>(7, 1);
        pt.z = corners.at<float>(7, 2);
        marker.points.push_back(pt);
        pt.x = corners.at<float>(3, 0);
        pt.y = corners.at<float>(3, 1);
        pt.z = corners.at<float>(3, 2);
        marker.points.push_back(pt);
        pt.x = corners.at<float>(6, 0);
        pt.y = corners.at<float>(6, 1);
        pt.z = corners.at<float>(6, 2);
        marker.points.push_back(pt);
        pt.x = corners.at<float>(4, 0);
        pt.y = corners.at<float>(4, 1);
        pt.z = corners.at<float>(4, 2);
        marker.points.push_back(pt);
        pt.x = corners.at<float>(5, 0);
        pt.y = corners.at<float>(5, 1);
        pt.z = corners.at<float>(5, 2);
        marker.points.push_back(pt);
        pt.x = corners.at<float>(0, 0);
        pt.y = corners.at<float>(0, 1);
        pt.z = corners.at<float>(0, 2);
        marker.points.push_back(pt);
        pt.x = corners.at<float>(1, 0);
        pt.y = corners.at<float>(1, 1);
        pt.z = corners.at<float>(1, 2);
        marker.points.push_back(pt);

        marker_array.markers.push_back(marker);
        marker_id++;
    }

    post_current_marker_id_ = marker_array.markers.size();
    for (int id_num_diff = post_num_prev_obj_marker_id_-post_current_marker_id_; id_num_diff > 0; id_num_diff--)
    {
        visualization_msgs::msg::Marker m;
        m.id = id_num_diff + post_current_marker_id_ - 1;
        m.action = m.DELETE;
        marker_array.markers.push_back(m);
    }
    post_num_prev_obj_marker_id_ = post_current_marker_id_; // For the next callback */
    pub_post_marker_->publish(marker_array);
}

}// namespace camera_detection_single_stage

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(camera_detection_single_stage::CameraDetectionSingleStageNode)