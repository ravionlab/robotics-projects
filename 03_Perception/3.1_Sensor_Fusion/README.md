# Sensor Fusion in Robotics

## Introduction
Sensor fusion is the process of combining data from multiple sensors to achieve more accurate, complete, and reliable information than what could be obtained from any individual sensor. In robotics, sensor fusion is essential for building robust perception systems that can operate in complex and dynamic environments.

## Theoretical Foundations

### Information Theory
- **Mutual Information**: Measuring the shared information between sensor data
- **Entropy**: Quantifying the uncertainty in sensor measurements
- **Information Gain**: Evaluating the value of additional sensor readings
- **Channel Capacity**: Understanding the limits of sensor data transmission

### Probability Theory
- **Bayes' Rule**: P(state|measurements) ∝ P(measurements|state) × P(state)
- **Joint Probability Distributions**: Modeling relationships between sensor data
- **Conditional Independence**: Simplifying fusion by identifying independent measurements
- **Markov Processes**: Modeling temporal dependencies in sequential measurements

### Estimation Theory
- **Maximum Likelihood Estimation**: Finding the most probable state
- **Bayesian Estimation**: Incorporating prior knowledge with sensor data
- **Minimum Mean Square Error (MMSE)**: Optimizing for average error
- **Maximum A Posteriori (MAP)**: Finding the most probable state given all evidence

## Sensor Fusion Architectures

### Hierarchical Fusion
```
                    High-Level Fusion
                           ↑
                    ┌──────┴──────┐
             Mid-Level Fusion  Mid-Level Fusion
                    ↑              ↑
             ┌──────┴──────┐ ┌────┴────┐
        Low-Level    Low-Level    Low-Level
        Fusion       Fusion       Fusion
          ↑            ↑            ↑
    ┌─────┴─────┐  ┌───┴───┐   ┌───┴───┐
Sensor A  Sensor B  Sensor C   Sensor D  Sensor E
```

### Centralized vs. Decentralized Fusion
- **Centralized**: All sensor data processed at a single point
  - Advantages: Global optimization, simpler implementation
  - Disadvantages: Single point of failure, communication bottlenecks
  
- **Decentralized**: Processing distributed across multiple nodes
  - Advantages: Robustness, scalability, reduced communication
  - Disadvantages: Suboptimal solutions, increased complexity

### Temporal Fusion Strategies
- **Sequential**: Process measurements one after another
- **Batch**: Process multiple measurements together
- **Sliding Window**: Process a moving window of recent measurements
- **Asynchronous**: Handle measurements with different arrival times and rates

## Common Fusion Algorithms

### Kalman Filter Family
- **Linear Kalman Filter**: Optimal for linear systems with Gaussian noise
  - Prediction step: x̂⁻ₖ = Fₖx̂ₖ₋₁ + Bₖuₖ, P⁻ₖ = FₖPₖ₋₁Fₖᵀ + Qₖ
  - Update step: Kₖ = P⁻ₖHₖᵀ(HₖP⁻ₖHₖᵀ + Rₖ)⁻¹, x̂ₖ = x̂⁻ₖ + Kₖ(zₖ - Hₖx̂⁻ₖ), Pₖ = (I - KₖHₖ)P⁻ₖ
  
- **Extended Kalman Filter (EKF)**: For nonlinear systems using linearization
  - Linearization through Jacobian matrices
  - Widely used in robot localization and SLAM

- **Unscented Kalman Filter (UKF)**: Better handling of nonlinearities
  - Uses sigma points to capture mean and covariance
  - More accurate than EKF for highly nonlinear systems

- **Information Filter**: Inverse covariance form of Kalman filter
  - Advantages for multi-sensor fusion and distributed implementation

### Particle Filters
- **Sequential Monte Carlo methods**: Represent state by particle set
- **Importance Sampling**: Weight particles based on sensor likelihoods
- **Resampling**: Concentrate particles in high-probability regions
- **Adaptive Particle Filters**: Adjust particle count based on uncertainty

### Bayesian Methods
- **Naive Bayes Fusion**: Assuming sensor independence
- **Bayesian Networks**: Modeling dependencies between sensors
- **Hidden Markov Models**: Sequential state estimation with discrete states
- **Dynamic Bayesian Networks**: Temporal extension of Bayesian networks

### Multi-Hypothesis Tracking
- **Track Initiation**: Creating new hypotheses for potential objects
- **Data Association**: Matching measurements to existing tracks
- **Hypothesis Management**: Pruning and merging tracking hypotheses
- **Joint Probabilistic Data Association**: Soft assignment of measurements

## Sensor-Specific Fusion Techniques

### Camera-LiDAR Fusion
- **Extrinsic Calibration**: Aligning coordinate frames between sensors
- **Point Cloud Colorization**: Adding visual information to 3D points
- **Depth-Guided Segmentation**: Using depth to improve image segmentation
- **3D Object Detection**: Combining 2D detections with 3D geometry

### IMU-GNSS Integration
- **Loosely Coupled**: Separate position and orientation solutions
- **Tightly Coupled**: Direct integration of raw measurements
- **Ultra-Tightly Coupled**: Integration at signal processing level
- **Error State Kalman Filter**: Common implementation approach

### Radar-Camera Fusion
- **Complementary Features**: Combining appearance and velocity information
- **Weather-Robust Perception**: Maintaining performance in adverse conditions
- **Long-Range Detection**: Extending perception range with radar
- **Cross-Modal Validation**: Confirming detections across sensor types

### Tactile-Vision Fusion
- **Grasp Refinement**: Using tactile feedback to adjust visual grasping
- **Surface Property Estimation**: Combining appearance and feel
- **Object Recognition**: Integrating visual and tactile features
- **Active Perception**: Guiding tactile exploration based on visual cues

## Implementation Considerations

### Temporal Alignment
- **Timestamp Synchronization**: Ensuring accurate measurement timing
- **Interpolation**: Estimating values between discrete measurements
- **Extrapolation**: Predicting values beyond available measurements
- **Buffering Strategies**: Managing measurement history

### Spatial Alignment
- **Sensor Calibration**: Determining relative poses between sensors
- **Registration Techniques**: Aligning data from different viewpoints
- **Feature Matching**: Establishing correspondences between sensor data
- **Coordinate Transformations**: Converting between reference frames

### Uncertainty Representation
- **Covariance Matrices**: For Gaussian uncertainty
- **Confidence Intervals**: For scalar values
- **Particle Sets**: For non-parametric distributions
- **Belief Functions**: For epistemic uncertainty

### Computational Efficiency
- **Real-time Constraints**: Meeting timing requirements
- **Parallel Processing**: Utilizing multi-core and GPU architectures
- **Approximate Algorithms**: Trading accuracy for speed
- **Sensor Selection**: Choosing which sensors to fuse based on context

## Software Implementation

### ROS Implementation
```python
#!/usr/bin/env python
import rospy
import numpy as np
from sensor_msgs.msg import Imu, NavSatFix
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovariance

class SensorFusion:
    def __init__(self):
        rospy.init_node('sensor_fusion_node')
        
        # Initialize state and covariance
        self.state = np.zeros(15)  # [position, velocity, orientation, accel_bias, gyro_bias]
        self.covariance = np.eye(15) * 0.1
        
        # Process and measurement noise
        self.Q = np.eye(15) * 0.01  # Process noise
        self.R_imu = np.eye(6) * 0.1  # IMU measurement noise
        self.R_gps = np.eye(3) * 2.0  # GPS measurement noise
        
        # Publishers and subscribers
        self.odom_pub = rospy.Publisher('fused_odometry', Odometry, queue_size=10)
        rospy.Subscriber('imu/data', Imu, self.imu_callback)
        rospy.Subscriber('gps/fix', NavSatFix, self.gps_callback)
        
        self.last_time = None
        
    def imu_callback(self, msg):
        current_time = rospy.Time.now().to_sec()
        if self.last_time is None:
            self.last_time = current_time
            return
            
        dt = current_time - self.last_time
        self.last_time = current_time
        
        # Extract IMU measurements
        accel = np.array([msg.linear_acceleration.x, 
                          msg.linear_acceleration.y, 
                          msg.linear_acceleration.z])
        gyro = np.array([msg.angular_velocity.x,
                         msg.angular_velocity.y,
                         msg.angular_velocity.z])
                         
        # Prediction step (simplified)
        self.predict(dt, accel, gyro)
        
        # Update step with IMU measurements
        self.update_imu(accel, gyro)
        
        # Publish result
        self.publish_state()
        
    def gps_callback(self, msg):
        # GPS update step
        gps_pos = np.array([msg.latitude, msg.longitude, msg.altitude])
        self.update_gps(gps_pos)
        
        # Publish updated state
        self.publish_state()
        
    def predict(self, dt, accel, gyro):
        # Simplified EKF prediction step
        # In a real implementation, this would include proper attitude integration
        # and nonlinear state transition
        
        # State transition matrix (simplified)
        F = np.eye(15)
        F[0:3, 3:6] = np.eye(3) * dt  # Position update from velocity
        
        # Predict state
        self.state = F @ self.state
        
        # Update position and velocity based on acceleration
        accel_corrected = accel - self.state[9:12]  # Apply accelerometer bias correction
        self.state[3:6] += accel_corrected * dt  # Update velocity
        self.state[0:3] += self.state[3:6] * dt + 0.5 * accel_corrected * dt**2  # Update position
        
        # Predict covariance
        self.covariance = F @ self.covariance @ F.T + self.Q
        
    def update_imu(self, accel, gyro):
        # Simplified IMU update
        # In a real implementation, this would be more complex
        
        # Measurement model (direct observation of acceleration and angular velocity)
        H = np.zeros((6, 15))
        H[0:3, 9:12] = np.eye(3)  # Accelerometer measures acceleration plus bias
        H[3:6, 12:15] = np.eye(3)  # Gyroscope measures angular velocity plus bias
        
        # Expected measurement
        z_expected = np.concatenate([self.state[9:12], self.state[12:15]])
        
        # Actual measurement
        z_actual = np.concatenate([accel, gyro])
        
        # Innovation
        y = z_actual - z_expected
        
        # Innovation covariance
        S = H @ self.covariance @ H.T + self.R_imu
        
        # Kalman gain
        K = self.covariance @ H.T @ np.linalg.inv(S)
        
        # Update state
        self.state = self.state + K @ y
        
        # Update covariance
        I = np.eye(15)
        self.covariance = (I - K @ H) @ self.covariance
        
    def update_gps(self, gps_pos):
        # Simplified GPS update
        # In a real implementation, would convert from geographic to local coordinates
        
        # Measurement model (direct observation of position)
        H = np.zeros((3, 15))
        H[0:3, 0:3] = np.eye(3)
        
        # Innovation
        y = gps_pos - self.state[0:3]
        
        # Innovation covariance
        S = H @ self.covariance @ H.T + self.R_gps
        
        # Kalman gain
        K = self.covariance @ H.T @ np.linalg.inv(S)
        
        # Update state
        self.state = self.state + K @ y
        
        # Update covariance
        I = np.eye(15)
        self.covariance = (I - K @ H) @ self.covariance
        
    def publish_state(self):
        # Create and publish odometry message
        odom_msg = Odometry()
        odom_msg.header.stamp = rospy.Time.now()
        odom_msg.header.frame_id = "map"
        
        # Fill position
        odom_msg.pose.pose.position.x = self.state[0]
        odom_msg.pose.pose.position.y = self.state[1]
        odom_msg.pose.pose.position.z = self.state[2]
        
        # Fill velocity
        odom_msg.twist.twist.linear.x = self.state[3]
        odom_msg.twist.twist.linear.y = self.state[4]
        odom_msg.twist.twist.linear.z = self.state[5]
        
        # Fill covariance (simplified)
        odom_msg.pose.covariance = [self.covariance[i, j] for i in range(6) for j in range(6)]
        
        # Publish
        self.odom_pub.publish(odom_msg)

if __name__ == '__main__':
    try:
        fusion = SensorFusion()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
```

## Performance Evaluation

### Metrics
- **Root Mean Square Error (RMSE)**: √(1/n ∑(x̂ᵢ - xᵢ)²)
- **Normalized Innovation Squared (NIS)**: yᵀS⁻¹y (chi-square distributed)
- **Consistency**: Agreement between estimated and actual error
- **Information Content**: Entropy reduction from sensor fusion

### Testing Methodology
- **Ground Truth Comparison**: Evaluating against known states
- **Cross-Validation**: Using one sensor to validate others
- **Monte Carlo Simulation**: Statistical performance over many runs
- **Sensitivity Analysis**: Robustness to sensor failures and degradation

### Benchmarking Datasets
- **KITTI**: Autonomous driving sensor suite
- **EuRoC**: Micro aerial vehicle datasets
- **TUM RGB-D**: Indoor robot navigation
- **Oxford RobotCar**: Long-term autonomy with diverse conditions

## Common Challenges and Solutions

### Sensor Misalignment
- **Cause**: Imperfect calibration or physical mounting
- **Effects**: Systematic errors in fusion results
- **Solutions**: Online calibration, robust estimation methods

### Asynchronous Measurements
- **Cause**: Different sensor update rates and timing
- **Effects**: Temporal inconsistency in fused data
- **Solutions**: Timestamp-based fusion, out-of-sequence measurement handling

### Conflicting Information
- **Cause**: Sensor disagreement or failure
- **Effects**: Degraded fusion performance
- **Solutions**: Outlier rejection, adaptive sensor weighting

### Computational Complexity
- **Cause**: High-dimensional state or many sensors
- **Effects**: Processing delays, reduced update rate
- **Solutions**: State dimension reduction, sensor selection, parallel processing

## Advanced Topics

### Deep Learning for Sensor Fusion
- **End-to-End Fusion**: Learning directly from raw sensor data to output
- **Feature-Level Fusion**: Learning joint representations across modalities
- **Uncertainty-Aware Networks**: Producing calibrated uncertainty estimates
- **Self-Supervised Fusion**: Learning without ground truth annotations

### Distributed Sensor Fusion
- **Consensus Algorithms**: Agreeing on state across multiple robots
- **Federated Kalman Filtering**: Distributing computation across nodes
- **Covariance Intersection**: Conservative fusion without cross-correlation knowledge
- **Distributed Particle Filters**: Sharing particles between processing nodes

### Active Sensor Fusion
- **Sensor Selection**: Choosing optimal sensors for current context
- **Information-Theoretic Planning**: Maximizing information gain
- **Adaptive Sampling**: Adjusting measurement rates based on needs
- **Sensor Placement Optimization**: Positioning sensors for optimal fusion

## Practical Exercises

### Exercise 1: IMU-Encoder Fusion
Implement a simple Kalman filter to fuse wheel encoder odometry with IMU measurements for improved robot localization.

### Exercise 2: Camera-LiDAR Calibration
Develop a procedure to determine the extrinsic calibration between a camera and LiDAR sensor.

### Exercise 3: Multi-Sensor Object Tracking
Implement a multi-hypothesis tracker that fuses detections from camera, LiDAR, and radar to track dynamic objects.

### Exercise 4: Sensor Fault Detection
Design an algorithm to detect and handle sensor failures in a fusion system using redundant information.

## Related Topics
- [SLAM](../3.2_SLAM/README.md): Using sensor fusion for mapping and localization
- [Computer Vision](../3.3_Computer_Vision/README.md): Visual perception techniques
- [Kalman Filters](../../02_Control_Systems/2.3_Kalman_Filters/README.md): State estimation theory 
