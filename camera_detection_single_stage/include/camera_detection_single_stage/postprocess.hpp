#pragma once
#include "camera_detection_single_stage/object.hpp"

namespace monoflex{

// Back project a 2D point x to 3D X given it's depth Z. Assume the camera is
// canonical K[I|0], K with 0 skew.
template <typename T>
inline void IBackprojectCanonical(const T *x, const T *K, T depth, T *X) {
    X[0] = (x[0] - K[2]) * depth * 1.0f/(K[0]);
    X[1] = (x[1] - K[5]) * depth * 1.0f/(K[4]);
    X[2] = depth;
}

// from apollo
struct TransformerParams {
  TransformerParams() { set_default(); }

  void set_default();

  int max_nr_iter;
  float learning_rate;
  float k_min_cost;
  float eps_cost;
};

class PostprocessCameraDetection
{
    
public:
    PostprocessCameraDetection(TransformerParams const& params);
    ~PostprocessCameraDetection() = default;
    bool transform(std::vector<BBOX>& bbox_vec, cv::Mat const& K, cv::Mat const& D);

    float CenterPointFromBbox(cv::Mat const& K, cv::Mat const& D, const float *bbox, const float *hwl, float ry, float *center);

    void ConstraintCenterPoint(const float *bbox, const float &z_ref, const float &ry, const float *hwl,
                                const float *k_mat, float* D, float *center, float *x, int height, int width);
    
    void UpdateCenterViaBackProjectZ(cv::Mat const& K, cv::Mat const& D, const float *bbox, const float& depth, float *center) const {

        float center_2d[2] = {(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2};

        cv::Point2f raw32 = cv::Point2f(center_2d[0], center_2d[1]), rect32;
        const cv::Mat src_pt(1, 1, CV_32FC2, &raw32.x);
        cv::Mat dst_pt(1, 1, CV_32FC2, &rect32.x);
        cv::undistortPoints(src_pt, dst_pt, K, D);
        center[0] = depth * rect32.x;
        center[1] = depth * rect32.y;
        center[2] = depth;
      }

private:
    TransformerParams params_;
};


// this is the inverse of the rotation matrix
template <typename T>
void GenRotMatrix(const T &ry, T *rot) {
    rot[0] = static_cast<T>(cos(ry));
    rot[2] = static_cast<T>(-sin(ry));
    rot[4] = static_cast<T>(1.0f);
    rot[6] = static_cast<T>(sin(ry));
    rot[8] = static_cast<T>(cos(ry));
    rot[1] = rot[3] = rot[5] = rot[7] = static_cast<T>(0);
}

template <typename T>
void GenCorners(T h, T w, T l, T *x_cor, T *y_cor, T *z_cor) {
    T half_w = static_cast<T>(w * 0.5f);
    T half_l = static_cast<T>(l * 0.5f);
    T half_h = static_cast<T>(h * 0.5f);
    y_cor[0] = y_cor[1] = y_cor[4] = y_cor[5] = -half_h;
    y_cor[2] = y_cor[3] = y_cor[6] = y_cor[7] = half_h;
    x_cor[0] = x_cor[5] = x_cor[6] = x_cor[7] = -half_w;
    x_cor[1] = x_cor[2] = x_cor[3] = x_cor[4] = half_w;
    z_cor[0] = z_cor[1] = z_cor[2] = z_cor[7] = -half_l;
    z_cor[4] = z_cor[5] = z_cor[3] = z_cor[6] = half_l;
    
}

// Multiply 3 x 3 matrix A with 3-dimensional vector x
template <typename T>
inline void IMultAx3x3(const T A[9], const T x[3], T Ax[3]) {
    T x0, x1, x2;
    x0 = x[0];
    x1 = x[1];
    x2 = x[2];
    Ax[0] = A[0] * x0 + A[1] * x1 + A[2] * x2;
    Ax[1] = A[3] * x0 + A[4] * x1 + A[5] * x2;
    Ax[2] = A[6] * x0 + A[7] * x1 + A[8] * x2;
}

template <typename T>
inline void IAdd3(const T x[3], T y[3]) {
  y[0] += x[0];
  y[1] += x[1];
  y[2] += x[2];
}

// Compute x=[R|t]*X, assuming R is 3x3 rotation matrix and t is a 3-vector.
template <typename T>
inline void IProjectThroughExtrinsic(const T *R, const T *t, const T *X, T *x) {
  IMultAx3x3(R, X, x);
  IAdd3(t, x);
}

// Compute x=K*X, assuming K is 3x3 upper triangular with K[8] = 1.0, do not
// consider radial distortion.
template <typename T>
inline void IProjectThroughIntrinsic(const T *K, const T *X, T *x) {
  x[0] = K[0] * X[0] + K[1] * X[1] + K[2] * X[2];
  x[1] = K[4] * X[1] + K[5] * X[2];
  x[2] = X[2];
}


template <typename T>
inline void IDistorsion(const T *D, const T *X, T *x) {
  T xp = X[0] / X[2];
  T yp = X[1] / X[2];
  x[2] = 1;

  T r2 = xp*xp + yp*yp;
  T r4 = r2*r2;
  T r6 = r4*r2;
  T a1 = 2*xp*yp;
  T k1 = D[0], k2 = D[1], p1 = D[2], p2 = D[3], k3 = D[4];
  T barrel_correction = 1 + k1*r2 + k2*r4 + k3*r6;
  x[0] = xp*barrel_correction + p1*a1 + p2*(r2+2*(xp*xp));
  x[1] = yp*barrel_correction + p1*(r2+2*(yp*yp)) + p2*a1;
}

// Compute x=K[R|t]*X, assuming K is 3x3 upper triangular with K[8] = 1.0,
// assuming R is 3x3 rotation matrix and t is a 3-vector, do not consider
// radial distortion.
template <typename T>
inline void IProjectThroughKRt(const T *K, T* D, const T *R, const T *t, const T *X, T *x) {
    T Rtx[3];
    IProjectThroughExtrinsic(R, t, X, Rtx);
    T tmp = Rtx[0];
    Rtx[0] = Rtx[2];
    Rtx[2] = tmp;
    T Dx[3];
    IDistorsion(D, Rtx, Dx);
    IProjectThroughIntrinsic(K, Dx, x);
}

}