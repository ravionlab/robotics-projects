# Wheeled Robots

## Introduction
Wheeled robots are among the most common mobile robot platforms due to their mechanical simplicity, energy efficiency, and reliable performance on flat terrain. This section explores the various aspects of wheeled robot locomotion, including kinematics, dynamics, control, and applications.

## Wheel Types and Configurations

### Standard Wheel Types
- **Fixed Wheels**: Rotates around its axle, fixed orientation
- **Steered Wheels**: Rotates around axle and can be reoriented
- **Castor Wheels**: Offset steering joint for automatic alignment
- **Swedish/Mecanum Wheels**: Rollers on perimeter for omnidirectional motion
- **Spherical/Ball Wheels**: Omnidirectional motion with actuation challenges

### Common Drive Configurations
- **Differential Drive**: Two independently controlled wheels (+ casters)
- **Ackermann Steering**: Car-like steering with front wheels
- **Tricycle Drive**: Single steerable drive wheel with two passive wheels
- **Synchro Drive**: All wheels steer and drive together
- **Omnidirectional Drive**: Using special wheels for holonomic motion
- **Skid Steering**: Tank-like steering through differential speeds
- **Tracked Configuration**: Using continuous tracks instead of wheels

### Configuration Comparison

| Configuration | Maneuverability | Mechanical Complexity | Control Complexity | Efficiency | Terrain Handling |
|---------------|-----------------|----------------------|-------------------|------------|-----------------|
| Differential | High | Low | Low | Medium | Low-Medium |
| Ackermann | Medium | Medium | Medium | High | Medium |
| Omnidirectional | Very High | High | High | Low | Low |
| Skid Steering | Medium | Low | Low | Low | High |
| Tracked | Medium | Medium | Low | Low | Very High |

## Kinematics of Wheeled Robots

### Kinematic Constraints
- **Rolling Constraint**: Wheel rolls without slipping
- **Sliding Constraint**: Wheel doesn't slide sideways
- **Non-holonomic Constraints**: Motion restrictions that cannot be integrated
- **Holonomic Systems**: No non-integrable constraints on motion

### Kinematic Models

#### Differential Drive Kinematics
The most common wheeled robot configuration with simple kinematics:

```
ẋ = (r/2) * (ωR + ωL) * cos(θ)
ẏ = (r/2) * (ωR + ωL) * sin(θ)
θ̇ = (r/L) * (ωR - ωL)

Where:
- (x,y,θ) is the robot pose
- r is the wheel radius
- L is the distance between wheels
- ωR and ωL are the angular velocities of the right and left wheels
```

```python
def differential_drive_kinematics(v_left, v_right, theta, wheel_radius, wheel_base, dt):
    """
    Forward kinematics for a differential drive robot.
    
    Args:
        v_left: Left wheel velocity (rad/s)
        v_right: Right wheel velocity (rad/s)
        theta: Current orientation (rad)
        wheel_radius: Radius of wheels (m)
        wheel_base: Distance between wheels (m)
        dt: Time step (s)
        
    Returns:
        New position (x, y) and orientation (theta)
    """
    import numpy as np
    
    # Linear and angular velocity
    v = wheel_radius * (v_right + v_left) / 2
    omega = wheel_radius * (v_right - v_left) / wheel_base
    
    # Update pose
    if abs(omega) < 1e-6:  # Moving in straight line
        x_dot = v * np.cos(theta)
        y_dot = v * np.sin(theta)
        theta_dot = 0
    else:  # Following an arc
        radius = v / omega
        x_dot = -radius * np.sin(theta) + radius * np.sin(theta + omega * dt)
        y_dot = radius * np.cos(theta) - radius * np.cos(theta + omega * dt)
        theta_dot = omega
    
    return x_dot, y_dot, theta_dot
```

#### Ackermann Steering Model
Used in car-like robots, with steering limited to the front wheels:

```
ẋ = v * cos(θ)
ẏ = v * sin(θ)
θ̇ = (v/L) * tan(φ)

Where:
- v is the linear velocity
- φ is the steering angle
- L is the wheelbase (distance between front and rear axles)
```

#### Omnidirectional Kinematics
For robots using mecanum or Swedish wheels, allowing motion in any direction:

```
[ẋ]   [cos(θ)  -sin(θ)  0] [vx]
[ẏ] = [sin(θ)   cos(θ)  0] [vy]
[θ̇]   [  0        0     1] [ω]

Where:
- (vx, vy, ω) are the robot's local frame velocities
```

### Inverse Kinematics
Converting desired robot motion to required wheel speeds:

```
Differential Drive:
ωR = (v + ω*L/2) / r
ωL = (v - ω*L/2) / r

Where:
- v is the desired linear velocity
- ω is the desired angular velocity
```

```python
def inverse_kinematics_differential_drive(v, omega, wheel_radius, wheel_base):
    """
    Inverse kinematics for differential drive robot.
    
    Args:
        v: Desired linear velocity (m/s)
        omega: Desired angular velocity (rad/s)
        wheel_radius: Radius of wheels (m)
        wheel_base: Distance between wheels (m)
        
    Returns:
        Required wheel velocities (v_left, v_right) in rad/s
    """
    # Calculate wheel velocities
    v_right = (2*v + omega*wheel_base) / (2*wheel_radius)
    v_left = (2*v - omega*wheel_base) / (2*wheel_radius)
    
    return v_left, v_right
```

## Dynamics and Control

### Dynamic Considerations
- **Inertia**: Resistance to changes in motion
- **Wheel Slip**: Loss of traction affecting motion
- **Motor Dynamics**: Torque-speed characteristics
- **Friction**: Rolling resistance and other friction forces
- **Load Distribution**: Effect on traction and stability

### Control Strategies

#### Motion Control
- **Path Following**: Keeping the robot on a predefined path
- **Trajectory Tracking**: Following a time-parameterized path
- **Point Stabilization**: Reaching and maintaining a goal position
- **Velocity Control**: Maintaining desired linear and angular velocities
- **Formation Control**: Maintaining relative positions with other robots

```python
def pid_motion_controller(current_pose, target_pose, gains):
    """
    PID controller for point stabilization.
    
    Args:
        current_pose: Current (x, y, theta) of the robot
        target_pose: Target (x, y, theta)
        gains: Dictionary with 'kp', 'ki', 'kd' gains for each component
        
    Returns:
        Control commands (v, omega)
    """
    import numpy as np
    
    # Extract positions
    x, y, theta = current_pose
    x_target, y_target, theta_target = target_pose
    
    # Calculate errors in global frame
    e_x = x_target - x
    e_y = y_target - y
    
    # Transform errors to robot frame
    e_x_robot = e_x * np.cos(theta) + e_y * np.sin(theta)
    e_y_robot = -e_x * np.sin(theta) + e_y * np.cos(theta)
    e_theta = np.arctan2(np.sin(theta_target - theta), np.cos(theta_target - theta))
    
    # PID control (simplified without integral and derivative terms)
    v = gains['kp']['linear'] * e_x_robot
    omega = gains['kp']['angular'] * e_theta + gains['kp']['lateral'] * e_y_robot
    
    return v, omega
```

#### Controllers for Wheeled Robots
- **PID Control**: Classic feedback control for tracking
- **Pure Pursuit**: Geometric path following approach
- **Dynamic Window Approach**: Local obstacle avoidance
- **Model Predictive Control**: Predicting future states for control
- **Nonlinear Control**: Advanced techniques for non-holonomic constraints

### Odometry and Localization
- **Wheel Encoders**: Measuring wheel rotations
- **Dead Reckoning**: Estimating position from wheel movement
- **Inertial Measurement**: Using IMUs for additional data
- **Sensor Fusion**: Combining multiple sensor inputs
- **Error Accumulation**: Dealing with drift in odometry

```python
def update_odometry(x, y, theta, d_left, d_right, wheel_base):
    """
    Update odometry based on wheel encoder readings.
    
    Args:
        x, y, theta: Current pose
        d_left, d_right: Distance traveled by each wheel
        wheel_base: Distance between wheels
        
    Returns:
        Updated pose (x, y, theta)
    """
    import numpy as np
    
    # Calculate linear and angular displacement
    d_center = (d_right + d_left) / 2
    d_theta = (d_right - d_left) / wheel_base
    
    # Update pose
    if abs(d_theta) < 1e-6:  # Moving in straight line
        x += d_center * np.cos(theta)
        y += d_center * np.sin(theta)
    else:  # Following an arc
        radius = d_center / d_theta
        x += radius * (np.sin(theta + d_theta) - np.sin(theta))
        y += radius * (np.cos(theta) - np.cos(theta + d_theta))
    
    theta = (theta + d_theta) % (2 * np.pi)
    
    return x, y, theta
```

## Navigation for Wheeled Robots

### Path Planning
- **Global Planning**: Finding paths through known environments
- **Local Planning**: Reactive navigation in the immediate vicinity
- **Obstacle Avoidance**: Strategies for detecting and avoiding collisions
- **Potential Fields**: Using repulsive and attractive forces for navigation
- **Sampling-Based Methods**: RRT, PRM for complex environments

### Mapping and SLAM
- **Occupancy Grid Maps**: Discretized representation of the environment
- **Feature-Based Maps**: Landmark identification and mapping
- **Topological Maps**: Graph-based environmental representation
- **SLAM Algorithms**: Simultaneous localization and mapping
- **Map Updates**: Maintaining and updating maps over time

### Mobile Robot Operating Systems
- **ROS Navigation Stack**: Comprehensive framework for wheeled robots
- **MoveBase**: Path planning and execution
- **AMCL**: Adaptive Monte Carlo Localization
- **Costmap2D**: Obstacle representation for planning
- **DWA Planner**: Dynamic Window Approach for local planning

## Design Considerations

### Mechanical Design
- **Chassis Structure**: Rigidity, weight, material considerations
- **Wheel Selection**: Size, material, tread pattern
- **Suspension Systems**: Shock absorption and terrain adaptation
- **Weight Distribution**: Center of gravity and stability
- **Environmental Protection**: Dust, water, temperature resistance

### Actuation and Power
- **Motor Types**: DC, brushless, stepper, servo
- **Motor Drivers**: H-bridges, ESCs, motor controllers
- **Gearing**: Torque vs. speed trade-offs
- **Battery Selection**: Capacity, weight, charging
- **Power Management**: Efficient use of available energy

### Sensing for Wheeled Robots
- **Proprioceptive Sensors**: Encoders, IMUs, current sensors
- **Exteroceptive Sensors**: LiDAR, cameras, ultrasonic sensors
- **Sensor Placement**: Coverage, protection, interference
- **Sensor Fusion**: Combining multiple sensor inputs
- **Filtering Techniques**: Kalman filters, particle filters

## Applications of Wheeled Robots

### Industrial Applications
- **Automated Guided Vehicles (AGVs)**: Material transport in factories
- **Autonomous Mobile Robots (AMRs)**: Flexible warehouse logistics
- **Inspection Robots**: Monitoring infrastructure and equipment
- **Agricultural Robots**: Autonomous tractors, harvesters, sprayers
- **Mining Robots**: Autonomous haulage and exploration

### Service Applications
- **Cleaning Robots**: Vacuum cleaners, floor scrubbers
- **Delivery Robots**: Last-mile delivery, food delivery
- **Security Robots**: Autonomous patrols and monitoring
- **Healthcare Robots**: Patient transport, supply delivery
- **Hospitality Robots**: Hotel and restaurant service robots

### Research and Exploration
- **Planetary Rovers**: Mars and lunar exploration
- **Search and Rescue**: Disaster response robots
- **Environmental Monitoring**: Data collection in remote areas
- **Research Platforms**: Test beds for algorithms and control strategies
- **Educational Robots**: Teaching programming and robotics

## Challenges and Limitations

### Wheeled Robot Limitations
- **Terrain Constraints**: Difficulty on rough, loose, or steep terrain
- **Step Climbing**: Limited by wheel diameter
- **Traction Issues**: Slippage on slick surfaces
- **Non-holonomic Constraints**: Motion planning complexity
- **Mechanical Wear**: Bearing and wheel maintenance

### Future Directions
- **Adaptive Wheels**: Morphing wheels to handle various terrain
- **Energy Efficiency**: Improved actuation and power management
- **Swarm Robotics**: Coordination of multiple wheeled robots
- **Hybrid Systems**: Combining wheeled locomotion with other modalities
- **Autonomous Navigation**: Improved perception and decision-making

## Practical Exercises

### Exercise 1: Differential Drive Implementation
Implement a basic differential drive robot model and control system in simulation.

### Exercise 2: Path Following
Design and implement a pure pursuit controller for path following.

### Exercise 3: Odometry Calibration
Develop a method to calibrate wheel encoders for accurate dead reckoning.

### Exercise 4: Obstacle Avoidance
Implement a reactive obstacle avoidance algorithm for a wheeled robot.

## Related Topics
- [Control Systems](../../02_Control_Systems/README.md): Control theory for wheeled robots
- [Perception](../../03_Perception/README.md): Sensing for navigation
- [Legged Robots](../5.2_Legged_Robots/README.md): Alternative locomotion approach
- [Reinforcement Learning](../../06_Reinforcement_Learning/README.md): Learning navigation policies 
