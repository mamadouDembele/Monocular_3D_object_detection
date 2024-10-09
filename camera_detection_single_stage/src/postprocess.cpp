#include "camera_detection_single_stage/postprocess.hpp"


namespace monoflex{

void TransformerParams::set_default() {
  max_nr_iter = 10;
  learning_rate = 0.7f;
  k_min_cost = 4 * sqrtf(2.0f);
  eps_cost = 1e-5f;
}

PostprocessCameraDetection::PostprocessCameraDetection(TransformerParams const& params) : params_(params)
{}

bool PostprocessCameraDetection::transform(std::vector<BBOX>& bbox_vec, cv::Mat const& K, cv::Mat const& D)
{
    if (bbox_vec.empty()) {return true;}
  
    int nr_transformed_obj = 0;
    const float PI = M_PI;
    for (auto &obj: bbox_vec) {
  
        // set object mapper options
        //float theta_ray = atan2(obj.xyz.x, obj.xyz.z);

        // process
        float object_center[3] = {obj.xyz.x, obj.xyz.y, obj.xyz.z};
        float dimension_hwl[3] = {obj.whl.y, obj.whl.x, obj.whl.z};
        float rotation_y = obj.theta;
    
        if (rotation_y < -PI) {
            rotation_y += 2 * PI;
        } else if (rotation_y >= PI) {
            rotation_y -= 2 * PI;
        }

        // adjust center point
        float bbox[4] = {0};
        bbox[0] = obj.box.x;
        bbox[1] = obj.box.y;
        bbox[2] = obj.box.x + obj.box.width;
        bbox[3] = obj.box.y + obj.box.height;

        CenterPointFromBbox(K, D, bbox, dimension_hwl, rotation_y, object_center);
        obj.xyz.x = object_center[0];
        obj.xyz.y = object_center[1];
        obj.xyz.z = object_center[2];

        ++nr_transformed_obj;
    }

    return nr_transformed_obj > 0;
}

float PostprocessCameraDetection::CenterPointFromBbox(cv::Mat const& K, cv::Mat const& D, const float *bbox, const float *hwl,
                                                            float ry, float *center) {
    float height_bbox = bbox[3] - bbox[1];
    float width_bbox = bbox[2] - bbox[0];
    if (width_bbox <= 0.0f || height_bbox <= 0.0f) {
        std::cerr << "Check predict bounding box, width or height is 0" << std::endl;
        return false;
    }

    float k_mat[9] = {0};
    for (size_t i = 0; i < 3; i++) {
        size_t i3 = i * 3;
        for (size_t j = 0; j < 3; j++) {k_mat[i3 + j] = K.at<float>(i, j);}
    }

    float f = (k_mat[0] + k_mat[4]) / 2;
    float depth = f * hwl[0]/height_bbox;

    // Compensate from the nearest vertical edge to center
    const float PI = M_PI;
    float theta_bbox = static_cast<float>(atan(hwl[1]/hwl[2]));
    float radius_bbox = std::sqrt(std::pow(hwl[2]/2, 2.0) + std::pow(hwl[1]/2, 2.0));

    float abs_ry = fabsf(ry);
    float theta_z = std::min(abs_ry, PI - abs_ry) + theta_bbox;
    theta_z = std::min(theta_z, PI - theta_z);
    depth += static_cast<float>(fabs(radius_bbox * sin(theta_z)));

    UpdateCenterViaBackProjectZ(K, D, bbox, depth, center);

    /* // Back-project to solve center
    float location[3] = {center[0], center[1], center[2]};
    IProjectThroughIntrinsic(k_mat, location, center_2d);
    center_2d[0] *= 1.0f/(center_2d[2]);
    center_2d[1] *= 1.0f/(center_2d[2]);
    ConstraintCenterPoint(bbox, depth, ry, hwl, k_mat, D, center, center_2d, height, width);
    if (fabsf(depth - center[2]) * 1.0f/(center[2]) > 0.1) {
        std::cout << "perform postprocess" << std::endl;
        
        //UpdateCenterViaBackProjectZ(bbox, hwl, center_2d, center, k_mat);
    } */

    return depth;
}


void PostprocessCameraDetection::ConstraintCenterPoint(const float *bbox, const float &z_ref, const float &ry, const float *hwl,
                                const float *k_mat, float* D, float *center, float *x, int height, int width)
{
    float center_2d_target[2] = {(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2};
    const float K_MIN_COST = params_.k_min_cost;
    const float EPS_COST_DELTA = params_.eps_cost;
    const float LEARNING_RATE = params_.learning_rate;
    const int MAX_ITERATION = params_.max_nr_iter;

    float cost_pre = 2.0f * static_cast<float>(width);
    float cost_delta = 0.0f;
    float center_temp[3] = {0};
    float rot_y[9] = {0};

    GenRotMatrix(ry, rot_y);

    int iter = 1;
    bool stop = false;
    float h = hwl[0];
    float w = hwl[1];
    float l = hwl[2];
    float x_corners[8] = {0};
    float y_corners[8] = {0};
    float z_corners[8] = {0};

    float x_upper_bound = static_cast<float>(width - 1);
    float y_upper_bound = static_cast<float>(height - 1);

    // Get dimension matrix
    /*
        l/2  l/2  -l/2  -l/2  l/2  l/2  -l/2  -l/2
    D =  0    0     0     0    -h   -h    -h    -h
        w/2  -w/2 -w/2   w/2  w/2  -w/2 -w/2  -w/2
    */
    GenCorners(h, w, l, x_corners, y_corners, z_corners);
    while (!stop) {
        // Back project 3D center from image x and depth z_ref to camera center_temp
        IBackprojectCanonical(x, k_mat, z_ref, center_temp);
        // From center to location
        //center_temp[1] += hwl[0] / 2;
        float x_min = std::numeric_limits<float>::max();
        float x_max = -std::numeric_limits<float>::max();
        float y_min = std::numeric_limits<float>::max();
        float y_max = -std::numeric_limits<float>::max();
        float x_images[3] = {0};

        for (int i = 0; i < 8; ++i) {
            // Bbox from x_images
            float x_box[3] = {x_corners[i], y_corners[i], z_corners[i]};
            IProjectThroughKRt(k_mat, D, rot_y, center_temp, x_box, x_images);
            x_images[0] *= 1.0f/(x_images[2]);
            x_images[1] *= 1.0f/(x_images[2]);
            x_min = std::min(x_min, x_images[0]);
            x_max = std::max(x_max, x_images[0]);
            y_min = std::min(y_min, x_images[1]);
            y_max = std::max(y_max, x_images[1]);
        }

        // Clamp bounding box from 0~boundary
        x_min = std::min(std::max(x_min, 0.0f), x_upper_bound);
        x_max = std::min(std::max(x_max, 0.0f), x_upper_bound);
        y_min = std::min(std::max(y_min, 0.0f), y_upper_bound);
        y_max = std::min(std::max(y_max, 0.0f), y_upper_bound);

        // Calculate 2D center point and get cost
        // cost = (center_gt - center_cal)**2
        float center_cur[2] = {(x_min + x_max) / 2, (y_min + y_max) / 2};
        float cost_cur = std::sqrt(std::pow(center_cur[0] - center_2d_target[0], 2.0) + std::pow(center_cur[1] - center_2d_target[1], 2.0));

        // Stop or continue
        if (cost_cur >= cost_pre) {
            stop = true;
        } else {
            memcpy(center, center_temp, sizeof(float) * 3);
            cost_delta = (cost_pre - cost_cur) / cost_pre;
            cost_pre = cost_cur;
            // Update 2D center point by descent method
            x[0] += (center_2d_target[0] - center_cur[0]) * LEARNING_RATE;
            x[1] += (center_2d_target[1] - center_cur[1]) * LEARNING_RATE;
            ++iter;
            // Termination condition
            stop = iter >= MAX_ITERATION || cost_delta < EPS_COST_DELTA ||
                    cost_pre < K_MIN_COST;
        }
    }
}

}
