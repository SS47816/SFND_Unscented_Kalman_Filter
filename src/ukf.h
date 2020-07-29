#ifndef UKF_H
#define UKF_H

#include <vector>
#include <iostream>
#include "Eigen/Dense"
#include "measurement_package.h"

class UKF {
 public:
  /**
   * Constructor
   */
  UKF();

  /**
   * Destructor
   */
  virtual ~UKF();

  /**
   * ProcessMeasurement
   * @param meas_package The latest measurement data of either radar or laser
   */
  void ProcessMeasurement(const MeasurementPackage& meas_package);

  /**
   * Prediction Predicts sigma points, the state, and the state covariance
   * matrix
   * @param delta_t Time between k and k+1 in s
   */
  void Prediction(double delta_t);

  /**
   * Updates the state and the state covariance matrix using a laser measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateLidar(const MeasurementPackage& meas_package);

  /**
   * Updates the state and the state covariance matrix using a radar measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateRadar(const MeasurementPackage& meas_package);


  // initially set to false, set to true in first call of ProcessMeasurement
  bool is_initialized_;

  // if this is false, laser measurements will be ignored (except for init)
  bool use_laser_;

  // if this is false, radar measurements will be ignored (except for init)
  bool use_radar_;

  // if this is false, the NIS won't be logged
  bool b_NIS_;

  // NIS threshold
  double NIS_lower_;
  double NIS_upper_;

  // Measurement count
  long lidar_count_;
  long radar_count_;

  // NIS outlier rate
  double NIS_lidar_lower_rate_;
  double NIS_lidar_upper_rate_;
  double NIS_radar_lower_rate_;
  double NIS_radar_upper_rate_;

  // // recording of the NIS
  // std::vector<double> NIS_radar_;
  // std::vector<double> NIS_lidar_;

  // state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
  Eigen::VectorXd x_;

  // state covariance matrix
  Eigen::MatrixXd P_;

  // // predicted radar measurement: [rho phi rho_d] in SI units and rad
  // Eigen::VectorXd z_radar_pred_;

  // // predicted lidar measurement: [pos_x pos_y] in SI units
  // Eigen::VectorXd z_lidar_pred_;

  // predicted sigma points in state space
  Eigen::MatrixXd Xsig_pred_;

  // predicted sigma points in augmented state space
  Eigen::MatrixXd Xsig_aug_;

  // Generate noise covariance matrix for Radar
  Eigen::MatrixXd R_radar_;

  // Generate noise covariance matrix for Lidar
  Eigen::MatrixXd R_lidar_;

  // time when the state is true, in us
  long long time_us_;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  double std_a_;

  // Process noise standard deviation yaw acceleration in rad/s^2
  double std_yawdd_;

  // Laser measurement noise standard deviation position1 in m
  double std_laspx_;

  // Laser measurement noise standard deviation position2 in m
  double std_laspy_;

  // Radar measurement noise standard deviation radius in m
  double std_radr_;

  // Radar measurement noise standard deviation angle in rad
  double std_radphi_;

  // Radar measurement noise standard deviation radius change in m/s
  double std_radrd_ ;

  // Weights of sigma points
  Eigen::VectorXd weights_;

  // State dimension
  int n_x_;

  // Augmented state dimension
  int n_aug_;

  // Sigma point spreading parameter
  double lambda_;

  // Radar measurement dimension
  int n_z_radar_;

  // Lidar measurement dimension
  int n_z_lidar_;

 private:
  void NormAngle(double& angle);

  void AugmentedSigmaPoints();

  void SigmaPointPrediction(const double delta_t);

  void PredictMeanAndCovariance();

};

#endif  // UKF_H