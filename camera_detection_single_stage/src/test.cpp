#include <iostream>
#include <iomanip>
#include "monoflex.hpp"
#include <filesystem>
#include <fstream>
#include <random>

int main()
{

    NET_PARAMS params;
    params.confidence_threshold = 0.35;
    params.model_path = "/home/mamadou/tmp/yolov8_onnx/monoflex_576_768_fp32_op.onnx";
    params.ep = EP::TENSORRT;
    params.precision = PRECISION::FP16;
    params.min_2d_height = 0.01;
    params.min_3d_height = 0.1;
    params.min_3d_width = 0.0;
    params.min_3d_length = 0.0;

    cv::Mat intrinsic_ = (cv::Mat_<float>(3, 3) << 1207.67742, 0.0, 992.05963, 0.0, 1210.81838, 603.34575, 0.0, 0.0, 1.0);
    MonoFlex3D monoflex(params);
    monoflex.setupCameraIntrinsec(intrinsic_);

    cv::VideoCapture cap("/home/mamadou/milla_shuttle4.mp4"); 
    
    // Check if camera opened successfully
    if(!cap.isOpened()){
        std::cout << "Error opening video stream or file" << std::endl;
        return -1;
    }

    float min_process_time(1000000000.0), max_process_time(0), avg_process_time(0), number_infer(0);
        
    while(1){
        cv::Mat frame;
        cap >> frame;

        auto start = std::chrono::system_clock::now();

        std::vector<BBOX> bbox_res;
        monoflex.RunSession(frame, bbox_res);
        monoflex.FilterByMinDims(bbox_res);
        cv::Mat out = frame.clone();
        for (auto const& b: bbox_res){
            std::cout << "[MONOFLEX]: " << b << std::endl;
            cv::Scalar color;
            std::string text;
            float probs = b.score;
            
            switch (b.label)
            {
            case LABELS::CAR:
				text = "car: " + std::to_string(probs);
                color = cv::Scalar(0, 0, 142);
                break;
            case LABELS::TRUCK:
				text = "truck: " + std::to_string(probs);
                color = cv::Scalar(0, 0, 70);
                break;
            case LABELS::BUS:
				text = "bus: " + std::to_string(probs);
                color = cv::Scalar(0, 60, 100);
                break;
            case LABELS::TRAILER:
				text = "trailer: " + std::to_string(probs);
                color = cv::Scalar(0, 0, 110);
                break;
            case LABELS::CONSTRUCTION_VEHICLE:
				text = "construction_vehicle: " + std::to_string(probs);
                color = cv::Scalar(0, 0, 70);
                break;
            case LABELS::PEDESTRIAN:
				text = "pedestrian: " + std::to_string(probs);
                color = cv::Scalar(220, 20, 60);
                break;
            case LABELS::MOTORCYCLE:
				text = "motorcycle: " + std::to_string(probs);
                color = cv::Scalar(0, 0, 230);
                break;
            case LABELS::BICYCLE:
				text = "bicycle: " + std::to_string(probs);
                color = cv::Scalar(119, 11, 32);
                break;
            case LABELS::TRAFFIC_CONE:
				text = "traffic_cone: " + std::to_string(probs);
                color = cv::Scalar(180,165,180);
                break;
            case LABELS::BARRIER:
				text = "barrier: " + std::to_string(probs);
                color = cv::Scalar(190,153,153);
                break;
            
            default:
                break;
            }
            cv::rectangle(out, b.box, color, 2);
            cv::putText(out, text, cv::Point2d(b.box.x, b.box.y -10), cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv::LINE_AA);

            float w = b.whl.x;
            float h = b.whl.y;
            float l = b.whl.z;
            float angle = b.theta;
            float x = b.xyz.x;
            float y = b.xyz.y;
            float z = b.xyz.z;

            cv::Mat corners = (cv::Mat_<float>(8, 3) << -w, -h, -l,
                                                        w, -h, -l,
                                                        w,  h, -l,
                                                        w,  h,  l,
                                                        w, -h,  l,
                                                        -w, -h,  l,
                                                        -w,  h,  l,
                                                        -w,  h, -l);
            corners = 0.5f * corners;
            float _cos = cosf(angle);
            float _sin = sinf(angle);

            // rotation of the eight
            for (int i = 0; i < 8; ++i) {
                corners.at<float>(i, 0) = corners.at<float>(i, 0) * _sin + corners.at<float>(i, 2) * _cos + x;
                corners.at<float>(i, 1) += y;
                corners.at<float>(i, 2) = -corners.at<float>(i, 2) * _sin + corners.at<float>(i, 0) * _cos + z;
            }

            std::vector<cv::Point2f> img_corners(8);

            for (int i = 0; i < 8; ++i) {
                float x = intrinsic_.at<float>(0, 0) * corners.at<float>(i, 0) + intrinsic_.at<float>(0, 1) *  corners.at<float>(i, 1) + intrinsic_.at<float>(0, 2) *  corners.at<float>(i, 2);
                float y = intrinsic_.at<float>(1, 1) * corners.at<float>(i, 1) + intrinsic_.at<float>(1, 2) *  corners.at<float>(i, 2);
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
        
        auto end = std::chrono::system_clock::now();

        std::ostringstream strs;
        double process_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
        strs << process_time;
        std::string text = "inference time: " + strs.str();
        cv::putText(out, text, cv::Point2d(100, 100), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2, cv::LINE_AA);

        number_infer = number_infer + 1.0;
        if (process_time < min_process_time){
            min_process_time = process_time;
        }
        else if (process_time > max_process_time){
            max_process_time = process_time;
        }
        avg_process_time += process_time;
        std::cout << "min process time: " << min_process_time << "ms";
        std::cout << " max process time: " << max_process_time << "ms";
        std::cout << " avg process time: " << avg_process_time/number_infer << "ms"<< std::endl; 


        if (out.empty())
        break;
    
        cv::imshow( "Frame", out);
    
        char c=(char)cv::waitKey(25);
        if(c==27)
        break;
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}