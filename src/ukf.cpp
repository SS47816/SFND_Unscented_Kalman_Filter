#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 30;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 30;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */

  // State dimension
  n_x_ = 5;

  // Augmented state dimension
  n_aug_ = n_x_ + 2;

  // Sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  // Radar measurement dimension
  n_z_radar_ = 3;

  // Lidar measurement dimension
  n_z_lidar_ = 2;

  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);

  // predicted sigma points in state space
  MatrixXd Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
  
  // predicted sigma points in augmented state space
  MatrixXd Xsig_aug_ = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  
  // measurement sigma points in state space
  MatrixXd Zsig_radar_ = MatrixXd(n_z_radar_, 2 * n_aug_ + 1);

  // measurement sigma points in state space
  MatrixXd Zsig_lidar_ = MatrixXd(n_z_lidar_, 2 * n_aug_ + 1);
  
  // Weights of sigma points
  weights_ = Eigen::VectorXd(2*n_aug_ + 1);
  weights_(0) = lambda_/(lambda_ + n_aug_);
  const double weight = 0.5/(lambda_ + n_aug_);
  for (int i = 1; i < 2*n_aug_+1; ++i) {  
    weights_(i) = weight;
  }

  // Generate noise covariance matrix for Radar
  R_radar_ = MatrixXd(n_z_radar_, n_z_radar_);
  R_radar_ << std_radr_ * std_radr_, 0.0, 0.0,
              0.0, std_radphi_ * std_radphi_, 0.0,
              0.0, 0.0, std_radrd_ * std_radrd_;

  // Generate noise covariance matrix for Lidar
  R_lidar_ = MatrixXd(n_z_lidar_, n_z_lidar_);
  R_lidar_ << std_laspx_ * std_laspx_, 0.0,
              0.0, std_laspy_ * std_laspy_;
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(const MeasurementPackage& meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */
  if (meas_package.sensor_type_ == MeasurementPackage::SensorType::RADAR)
  {
    const double rho = meas_package.raw_measurements_(0);
    const double phi = meas_package.raw_measurements_(1);
    const double rho_d = meas_package.raw_measurements_(2);

    const double x = rho*std::cos(phi);
    const double y = rho*std::sin(phi);

    // update the state vector
    x_ << x, y, rho_d, phi, 0.0;

    // update the state covariance matrix
    P_ << std_radr_ * std_radr_, 0.0, 0.0, 0.0, 0.0,
          0.0, std_radr_ * std_radr_, 0.0, 0.0, 0.0,
          0.0, 0.0, std_radrd_ * std_radrd_, 0.0, 0.0,
          0.0, 0.0, 0.0, std_radphi_ * std_radphi_, 0.0,
          0.0, 0.0, 0.0, 0.0, 1;
  }
  else if (meas_package.sensor_type_ == MeasurementPackage::SensorType::LASER)
  {
    const double x = meas_package.raw_measurements_(0);
    const double y = meas_package.raw_measurements_(1);

    // update the state vector
    x_ << x, y, 0.0, 0.0, 0.0;

    // update the state covariance matrix
    P_ << std_laspx_ * std_laspx_, 0.0, 0.0, 0.0, 0.0,
          0.0, std_laspy_ * std_laspy_, 0.0, 0.0, 0.0,
          0.0, 0.0, 1, 0.0, 0.0,
          0.0, 0.0, 0.0, 1, 0.0,
          0.0, 0.0, 0.0, 0.0, 1;
  }

  is_initialized_ = true;
  time_us_ = meas_package.timestamp_;
}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */
}

void UKF::UpdateLidar(const MeasurementPackage& meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
}

void UKF::UpdateRadar(const MeasurementPackage& meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
}