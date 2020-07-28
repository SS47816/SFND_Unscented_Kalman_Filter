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

  // if this is false, the NIS won't be logged
  b_NIS_ = true;

  // NIS threshold
  NIS_lower_ = 0.352;
  NIS_upper_ = 7.815;

  // Measurement count
  lidar_count_ = 0;
  radar_count_ = 0;

  // NIS outlier rate
  NIS_lidar_lower_rate_ = 0.0;
  NIS_lidar_upper_rate_ = 0.0;
  NIS_radar_lower_rate_ = 0.0;
  NIS_radar_upper_rate_ = 0.0;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 3;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 3;
  
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

  // initialize state vector
  x_ = VectorXd(n_x_);

  // initialize augmented state vector
  x_aug_ = VectorXd(n_aug_);

  // initialize covariance matrix
  P_ = MatrixXd(n_x_, n_x_);

  // initialize augmented covariance matrix
  P_aug_ = MatrixXd(n_aug_, n_aug_);

  // Radar measurement dimension
  n_z_radar_ = 3;

  // Lidar measurement dimension
  n_z_lidar_ = 2;

  // initialize predicted radar measurement
  z_radar_pred_ = VectorXd(n_z_radar_);

  // initialize predicted radar measurement
  z_lidar_pred_ = VectorXd(n_z_lidar_);

  // initial predicted radar measurement covariance matrix
  S_radar_pred_ = MatrixXd(n_z_radar_, n_z_radar_);

  // initial predicted lidar measurement covariance matrix
  S_lidar_pred_ = MatrixXd(n_z_lidar_, n_z_lidar_);

  // predicted sigma points in state space
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
  
  // predicted sigma points in augmented state space
  Xsig_aug_ = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  
  // measurement sigma points in state space
  Zsig_radar_ = MatrixXd(n_z_radar_, 2 * n_aug_ + 1);

  // measurement sigma points in state space
  Zsig_lidar_ = MatrixXd(n_z_lidar_, 2 * n_aug_ + 1);
  
  // Weights of sigma points
  weights_ = VectorXd(2*n_aug_ + 1);
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

void UKF::NormAngle(double& angle)
{
  while (angle > M_PI) angle -= 2.0 * M_PI;
  while (angle < -M_PI) angle += 2.0 * M_PI;
}

void UKF::AugmentedSigmaPoints()
{
  // create augmented mean state
  x_aug_.head(n_x_) = x_;
  x_aug_(n_x_) = 0.0;
  x_aug_(n_x_+1) = 0.0;

  // create augmented covariance matrix
  P_aug_.fill(0.0);
  P_aug_.topLeftCorner(n_x_, n_x_) = P_;
  P_aug_(n_x_, n_x_) = std_a_*std_a_;
  P_aug_(n_x_+1, n_x_+1) = std_yawdd_*std_yawdd_;

  // create square root matrix
  MatrixXd L = P_aug_.llt().matrixL();

  // create augmented sigma points
  Xsig_aug_.fill(0.0);
  Xsig_aug_.col(0)  = x_aug_;
  for (int i = 0; i< n_aug_; ++i)
  {
    Xsig_aug_.col(i+1) = x_aug_ + sqrt(lambda_+n_aug_) * L.col(i);
    Xsig_aug_.col(i+1+n_aug_) = x_aug_ - sqrt(lambda_+n_aug_) * L.col(i);
  }
}

void UKF::SigmaPointPrediction(const double delta_t)
{
  // predict sigma points
  for (int i = 0; i< 2*n_aug_+1; ++i) {
    // extract values for better readability
    const double p_x = Xsig_aug_(0, i);
    const double p_y = Xsig_aug_(1, i);
    const double v = Xsig_aug_(2, i);
    const double yaw = Xsig_aug_(3, i);
    const double yawd = Xsig_aug_(4, i);
    const double nu_a = Xsig_aug_(5, i);
    const double nu_yawdd = Xsig_aug_(6, i);

    // predicted state values
    double px_p, py_p;

    // avoid division by zero
    if (fabs(yawd) > std::numeric_limits<double>::epsilon()) {
        px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
        py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
    } else {
        px_p = p_x + v*delta_t*cos(yaw);
        py_p = p_y + v*delta_t*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    // add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    // write predicted sigma point into right column
    Xsig_pred_(0, i) = px_p;
    Xsig_pred_(1, i) = py_p;
    Xsig_pred_(2, i) = v_p;
    Xsig_pred_(3, i) = yaw_p;
    Xsig_pred_(4, i) = yawd_p;
  }
}

void UKF::PredictMeanAndCovariance()
{
  // predicted state mean
  VectorXd x_pred = VectorXd(n_x_);
  x_pred.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // iterate over sigma points
    x_pred = x_pred + weights_(i) * Xsig_pred_.col(i);
  }
  x_ = std::move(x_pred);

  // predicted state covariance matrix
  MatrixXd P_pred = MatrixXd(n_x_, n_x_);
  P_pred.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // iterate over sigma points
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    NormAngle(x_diff(3));
    P_ = P_ + weights_(i) * x_diff * x_diff.transpose() ;
  }
  P_ = std::move(P_pred);
}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */
  
  AugmentedSigmaPoints();

  SigmaPointPrediction(delta_t);
  
  PredictMeanAndCovariance();
}

void UKF::UpdateLidar(const MeasurementPackage& meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */

  // transform sigma points to lidar space
  for (int i = 0; i < Xsig_pred_.cols(); i++)
  {
    // Read needed sigma point info
    Zsig_lidar_(0, i) = Xsig_pred_(0, i);
    Zsig_lidar_(1, i) = Xsig_pred_(1, i);
  }

  // update the predicted state
  VectorXd z_pred = VectorXd(n_z_lidar_);
  z_pred.fill(0.0);
  for (int i = 0; i < weights_.size(); ++i)
  {
    z_pred += weights_(i) * Zsig_lidar_.col(i);
  }
  z_lidar_pred_ = z_pred;


  // update the predicted covariance
  MatrixXd S_lidar_pred = MatrixXd(n_z_lidar_, n_z_lidar_);
  S_lidar_pred.fill(0.0);
  VectorXd z_diff;
  for (int i = 0; i < weights_.size(); i++)
  {
    z_diff = Zsig_lidar_.col(i) - z_lidar_pred_;
    S_lidar_pred += weights_(i) * (z_diff) * (z_diff).transpose();
  }

  // Add measurement noise
  S_lidar_pred += R_lidar_;
  S_lidar_pred_ = S_lidar_pred;

  MatrixXd T_XZ = MatrixXd(n_x_, n_z_lidar_); // Cross-correlation matrix
  VectorXd x_diff = VectorXd(n_x_);

  // Calculate Cross-correlation matrix
  T_XZ.fill(0.0);
  for (int i = 0; i < weights_.size(); i++)
  {
    x_diff = Xsig_pred_.col(i) - x_;
    T_XZ += weights_(i) * (x_diff) * (z_diff).transpose();
  }

  // Calculate Kalman gain K
  MatrixXd K = T_XZ * S_lidar_pred_.inverse();

  // Update state
  z_diff = meas_package.raw_measurements_ - z_lidar_pred_;
  x_ += K * z_diff;

  // Update covariance matrix
  P_ -= K * S_lidar_pred_ * K.transpose();
  
  if (b_NIS_)
  {
    const double nis = z_diff.transpose() * S_lidar_pred_.inverse() * z_diff;
    if (nis <= NIS_lower_)
    {
      NIS_lidar_lower_rate_ = (NIS_lidar_lower_rate_*lidar_count_ + 1)/(++lidar_count_);
    }
    else if (nis >= NIS_upper_)
    {
      NIS_lidar_upper_rate_ = (NIS_lidar_upper_rate_*lidar_count_ + 1)/(++lidar_count_);
    }
    
    std::cout << "NIS_lidar_lower_rate_: " << NIS_lidar_lower_rate_ << std::endl;
    std::cout << "NIS_lidar_upper_rate_: " << NIS_lidar_upper_rate_ << std::endl;
  }
}

void UKF::UpdateRadar(const MeasurementPackage& meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */

  // transform sigma points to radar space
  for (int i = 0; i < Xsig_pred_.cols(); i++)
  {
    const double x = Xsig_pred_(0, i);
    const double y = Xsig_pred_(1, i);
    const double vel_abs = Xsig_pred_(2, i);
    const double yaw_angle = Xsig_pred_(3, i);

    const double v_x = vel_abs * std::cos(yaw_angle);
    const double v_y = vel_abs * std::sin(yaw_angle);

    // Transform sigma point into measurement space
    Zsig_radar_(0, i) = std::sqrt(x * x + y * y);
    Zsig_radar_(1, i) = std::atan2(y, x);
    Zsig_radar_(2, i) = (x * v_x + y * v_y) / std::sqrt(x * x + y * y);
  }

  // update the predicted state
  VectorXd z_pred = VectorXd(n_z_radar_);
  z_pred.fill(0.0);
  for (int i = 0; i < weights_.size(); ++i)
  {
    z_pred += weights_(i) * Zsig_radar_.col(i);
  }
  z_radar_pred_ = z_pred;

  // update the predicted covariance
  MatrixXd S_radar_pred = MatrixXd(n_z_radar_, n_z_radar_);
  S_radar_pred.fill(0.0);
  VectorXd z_diff;
  for (int i = 0; i < weights_.size(); i++)
  {
    z_diff = Zsig_radar_.col(i) - z_radar_pred_;
    NormAngle(z_diff(1));
    S_radar_pred += weights_(i) * (z_diff) * (z_diff).transpose();
  }

  // Add measurement noise
  S_radar_pred += R_radar_;
  S_radar_pred_ = S_radar_pred;

  MatrixXd T_XZ = MatrixXd(n_x_, n_z_radar_); // Cross-correlation matrix
  VectorXd x_diff = VectorXd(n_x_);

  // Calculate Cross-correlation matrix
  T_XZ.fill(0.0);
  for (int i = 0; i < weights_.size(); i++)
  {
    x_diff = Xsig_pred_.col(i) - x_;
    NormAngle(x_diff(3));

    T_XZ += weights_(i) * (x_diff) * (z_diff).transpose();
  }

  // Calculate Kalman gain K
  MatrixXd K = T_XZ * S_radar_pred_.inverse();

  // Update state
  z_diff = meas_package.raw_measurements_ - z_radar_pred_;
  NormAngle(z_diff(1));
  x_ += K * z_diff;

  // Update covariance matrix
  P_ -= K * S_radar_pred_ * K.transpose();
  
  if (b_NIS_)
  {
    const double nis = z_diff.transpose() * S_radar_pred_.inverse() * z_diff;
    if (nis <= NIS_lower_)
    {
      NIS_radar_lower_rate_ = (NIS_radar_lower_rate_*lidar_count_ + 1)/(++lidar_count_);
    }
    else if (nis >= NIS_upper_)
    {
      NIS_radar_upper_rate_ = (NIS_radar_upper_rate_*lidar_count_ + 1)/(++lidar_count_);
    }
    std::cout << "NIS_radar_lower_rate_: " << NIS_radar_lower_rate_ << std::endl;
    std::cout << "NIS_radar_upper_rate_: " << NIS_radar_upper_rate_ << std::endl;
  }
}

void UKF::ProcessMeasurement(const MeasurementPackage& meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */

  if (!is_initialized_)
  {
    std::cout << "UKF initializing..." << std::endl;
    // initialize UKF based on the sensor type
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

      is_initialized_ = true;
      time_us_ = meas_package.timestamp_;
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

      is_initialized_ = true;
      time_us_ = meas_package.timestamp_;
    }

    std::cout << "UKF initialized" << std::endl;
  }
  else // if UKF is initialized
  {
    // update the time interval
    const double dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
    time_us_ = meas_package.timestamp_;
    
    // Prediction Step
    Prediction(dt);
    
    // Update Step
    if (meas_package.sensor_type_ == MeasurementPackage::SensorType::RADAR)
    {
      UpdateRadar(meas_package);
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::SensorType::LASER)
    {
      UpdateLidar(meas_package);
    }

    // std::cout << "UKF updated" << std::endl;
  }
}