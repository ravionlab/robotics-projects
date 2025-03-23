# Robot Kinematics

## Introduction
Kinematics is the branch of robotics that studies the motion of robotic mechanisms without considering the forces or moments that cause the motion. It forms the mathematical foundation for understanding, designing, and controlling robotic manipulators. This section explores forward and inverse kinematics, velocity kinematics, and workspace analysis for robotic systems.

## Coordinate Systems and Transformations

### Coordinate Frames
- **World Frame**: Global reference frame for the robot's environment
- **Base Frame**: Fixed to the robot's base
- **Link Frames**: Attached to each link of the manipulator
- **End-Effector Frame**: Attached to the end-effector or tool
- **Object Frames**: Attached to objects of interest

### Homogeneous Transformations
- **Translation Vector**: Position component of transformation
- **Rotation Matrix**: Orientation component of transformation
- **Homogeneous Matrix**: Combined 4×4 transformation matrix
- **Composition of Transformations**: Multiplying transformation matrices
- **Inverse Transformations**: Reversing transformations

```
T = [R  p]
    [0  1]

where:
- T is the 4×4 homogeneous transformation matrix
- R is the 3×3 rotation matrix
- p is the 3×1 translation vector
```

### Rotation Representations
- **Rotation Matrices**: 3×3 orthogonal matrices
- **Euler Angles**: Roll, pitch, yaw or other sequences
- **Axis-Angle**: Rotation around a single axis
- **Quaternions**: Four-parameter representation
- **Converting Between Representations**: Handling singularities

## Forward Kinematics

### Denavit-Hartenberg (DH) Convention
- **DH Parameters**: Joint angle (θ), link length (a), link offset (d), link twist (α)
- **Parameter Assignment**: Systematic method for frame attachment
- **DH Transformation Matrix**: Standard form using four parameters
- **Modified DH Convention**: Alternative parameter definitions

```
DH Transformation Matrix:
T_i-1,i = [cos(θ_i)  -sin(θ_i)cos(α_i)   sin(θ_i)sin(α_i)   a_i*cos(θ_i)]
          [sin(θ_i)   cos(θ_i)cos(α_i)  -cos(θ_i)sin(α_i)   a_i*sin(θ_i)]
          [    0          sin(α_i)           cos(α_i)            d_i     ]
          [    0             0                  0                 1       ]
```

### Forward Kinematics Algorithm
1. Assign coordinate frames to links following DH convention
2. Determine DH parameters for each joint
3. Compute individual transformation matrices for each joint
4. Multiply transformations to get end-effector pose

```python
def forward_kinematics(dh_params, joint_values):
    """
    Compute forward kinematics using DH parameters.
    
    Args:
        dh_params: List of DH parameters [a, alpha, d, theta_offset]
        joint_values: List of joint positions
        
    Returns:
        Homogeneous transformation matrix for end-effector
    """
    import numpy as np
    
    T = np.eye(4)  # Identity matrix
    
    for i, theta in enumerate(joint_values):
        a, alpha, d, theta_offset = dh_params[i]
        
        # Apply joint value + offset for revolute joints
        theta = theta + theta_offset
        
        # DH transformation matrix
        T_i = np.array([
            [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
            [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
            [0, np.sin(alpha), np.cos(alpha), d],
            [0, 0, 0, 1]
        ])
        
        # Update transformation
        T = T @ T_i
    
    return T
```

### Product of Exponentials (POE) Formula
- **Screw Theory Basics**: Twists and exponential coordinates
- **Joint Screws**: Representing joint motion
- **POE Formula**: Alternative to DH for forward kinematics
- **Advantages**: More geometric interpretation, no frame assignments

## Inverse Kinematics

### Problem Statement
- **Definition**: Finding joint angles for a desired end-effector pose
- **Challenges**: Multiple solutions, singularities, no solution cases
- **Redundancy**: More joints than needed for task
- **Optimization Criteria**: Selecting among multiple solutions

### Analytical Solutions
- **Closed-Form Solutions**: Direct algebraic approach
- **Geometric Approach**: Using manipulator geometry
- **Decoupling**: Separating position and orientation
- **Arm and Wrist Partitioning**: For 6-DOF manipulators
- **Pieper's Method**: For specific joint configurations

```python
def analytical_inverse_kinematics_6dof(T_target):
    """
    Analytical inverse kinematics for a specific 6-DOF robot with
    a spherical wrist configuration.
    
    Args:
        T_target: Target homogeneous transformation matrix
        
    Returns:
        List of possible joint configurations
    """
    import numpy as np
    
    # Extract position and rotation
    R = T_target[0:3, 0:3]
    p = T_target[0:3, 3]
    
    # Robot specific parameters
    d1 = 0.1  # Base to shoulder height
    a2 = 0.5  # Upper arm length
    a3 = 0.4  # Forearm length
    d4 = 0.0  # Wrist height
    
    # Wrist center calculation
    wrist_center = p - d4 * R @ np.array([0, 0, 1])
    
    # Solve for first joint (base rotation)
    theta1 = np.arctan2(wrist_center[1], wrist_center[0])
    
    # Solve for joints 2 and 3 using law of cosines
    r = np.sqrt(wrist_center[0]**2 + wrist_center[1]**2)
    s = wrist_center[2] - d1
    D = (r**2 + s**2 - a2**2 - a3**2) / (2 * a2 * a3)
    
    # Check if target is reachable
    if abs(D) > 1:
        raise ValueError("Target position out of reach")
    
    # Elbow up and down solutions
    theta3_up = np.arccos(D)
    theta3_down = -np.arccos(D)
    
    # Solve for theta2 in both cases
    theta2_up = np.arctan2(s, r) - np.arctan2(a3 * np.sin(theta3_up), a2 + a3 * np.cos(theta3_up))
    theta2_down = np.arctan2(s, r) - np.arctan2(a3 * np.sin(theta3_down), a2 + a3 * np.cos(theta3_down))
    
    # Calculate rotation matrices for the first 3 joints
    R03_up = calculate_R03(theta1, theta2_up, theta3_up)  # Function to calculate R03
    R03_down = calculate_R03(theta1, theta2_down, theta3_down)
    
    # Solve for wrist joints (4, 5, 6)
    wrist_up = solve_wrist_angles(R03_up.T @ R)  # Function to solve wrist angles
    wrist_down = solve_wrist_angles(R03_down.T @ R)
    
    # Combine solutions
    solutions = [
        [theta1, theta2_up, theta3_up, *wrist_up],
        [theta1, theta2_down, theta3_down, *wrist_down]
    ]
    
    return solutions
```

### Numerical Solutions
- **Jacobian-Based Methods**: Iterative approaches
- **Newton-Raphson**: Solving nonlinear equations
- **Cyclic Coordinate Descent**: Iteratively adjusting one joint
- **Optimization Techniques**: Gradient descent, CCD, quasi-Newton
- **Singularity-Robust Methods**: Handling near-singular cases

```python
def numerical_inverse_kinematics(target_pose, initial_guess, dh_params, max_iterations=100, tolerance=1e-6):
    """
    Numerical inverse kinematics using the Jacobian pseudoinverse method.
    
    Args:
        target_pose: Target end-effector pose as homogeneous transformation
        initial_guess: Initial joint values
        dh_params: DH parameters of the robot
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance
        
    Returns:
        Joint values that achieve the target pose
    """
    import numpy as np
    
    joint_values = np.array(initial_guess)
    
    for i in range(max_iterations):
        # Forward kinematics to get current pose
        current_pose = forward_kinematics(dh_params, joint_values)
        
        # Compute pose error
        pos_error = target_pose[0:3, 3] - current_pose[0:3, 3]
        
        # Convert rotation matrices to axis-angle representation for orientation error
        # (simplified - in practice, you would use a more robust method)
        R_error = np.dot(current_pose[0:3, 0:3].T, target_pose[0:3, 0:3])
        angle, axis = rotation_matrix_to_axis_angle(R_error)  # Function to extract angle and axis
        ori_error = angle * axis
        
        # Combine errors
        error = np.concatenate([pos_error, ori_error])
        error_norm = np.linalg.norm(error)
        
        # Check convergence
        if error_norm < tolerance:
            break
            
        # Compute Jacobian
        J = calculate_jacobian(joint_values, dh_params)  # Function to compute Jacobian
        
        # Compute pseudoinverse
        J_pinv = np.linalg.pinv(J)
        
        # Update joint values
        joint_values = joint_values + np.dot(J_pinv, error)
        
        # Apply joint limits (if needed)
        # joint_values = np.clip(joint_values, lower_limits, upper_limits)
    
    return joint_values
```

## Velocity Kinematics

### The Jacobian Matrix
- **Definition**: Relates joint velocities to end-effector velocities
- **Analytical Jacobian**: Using partial derivatives
- **Geometric Jacobian**: Using screw theory
- **Jacobian Properties**: Rank, condition number, manipulability
- **Calculating the Jacobian**:
  - J = ∂x/∂q where x is end-effector pose and q is joint variables

```
Geometric Jacobian Structure:
J = [Jv]  where Jv relates joint velocities to linear velocity
    [Jω]        Jω relates joint velocities to angular velocity
```

### Differential Kinematics
- **Linear Approximation**: ∆x = J·∆q
- **Velocity Relationship**: ẋ = J·q̇
- **Acceleration Relationship**: ẍ = J·q̈ + J̇·q̇
- **Force/Torque Relationship**: τ = JᵀF
- **Small Displacement Analysis**: Computing small motions

```python
def calculate_geometric_jacobian(joint_values, dh_params):
    """
    Calculate the geometric Jacobian matrix.
    
    Args:
        joint_values: Current joint positions
        dh_params: DH parameters of the robot
        
    Returns:
        Geometric Jacobian matrix
    """
    import numpy as np
    
    n_joints = len(joint_values)
    J = np.zeros((6, n_joints))
    
    # Forward kinematics to get current transformations
    T_current = forward_kinematics(dh_params, joint_values)
    
    # End-effector position
    p_end = T_current[0:3, 3]
    
    # Compute transformation to each joint
    T = np.eye(4)
    z_prev = np.array([0, 0, 1])  # Initial z-axis
    
    for i in range(n_joints):
        a, alpha, d, theta_offset = dh_params[i]
        theta = joint_values[i] + theta_offset
        
        # DH transformation for this joint
        T_i = np.array([
            [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
            [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
            [0, np.sin(alpha), np.cos(alpha), d],
            [0, 0, 0, 1]
        ])
        
        T = T @ T_i
        
        # Joint position
        p_i = T[0:3, 3]
        
        # Joint axis
        z_i = T[0:3, 0:3] @ np.array([0, 0, 1])
        
        # Jacobian columns
        J[0:3, i] = np.cross(z_i, p_end - p_i)  # Linear velocity component
        J[3:6, i] = z_i  # Angular velocity component
    
    return J
```

### Singularity Analysis
- **Definition**: Configurations where Jacobian loses rank
- **Types of Singularities**:
  - Boundary singularities: At workspace boundaries
  - Internal singularities: Inside workspace
  - Algorithmic singularities: Due to representation
- **Identifying Singularities**: Determinant of Jacobian
- **Singularity-Robust Control**: Methods to handle singularities

### Manipulability
- **Manipulability Measure**: w = √det(J·Jᵀ)
- **Manipulability Ellipsoid**: Visualization of motion capabilities
- **Task-Oriented Measures**: For specific motion directions
- **Condition Number**: Ratio of maximum to minimum singular values
- **Optimizing Manipulability**: Configurations with best performance

## Redundancy Resolution

### Redundant Manipulators
- **Definition**: More DOF than required for the task
- **Self-Motion**: Joint movement that doesn't affect end-effector pose
- **Null Space**: Space of joint velocities that produce no end-effector motion
- **Secondary Tasks**: Utilizing redundancy for additional objectives

### Redundancy Resolution Methods
- **Pseudoinverse Solutions**: Minimum norm solution
- **Weighted Pseudoinverse**: Prioritizing certain joints
- **Null Space Projection**: q̇ = J⁺ẋ + (I - J⁺J)q̇₀
- **Task Priority Framework**: Hierarchical task organization
- **Optimization-Based Approaches**: Quadratic programming formulations

```python
def redundancy_resolution(J, x_dot, q_dot_0):
    """
    Resolve redundancy using the null space projection method.
    
    Args:
        J: Jacobian matrix
        x_dot: Desired end-effector velocity
        q_dot_0: Secondary task joint velocity
        
    Returns:
        Joint velocities that achieve primary and secondary tasks
    """
    import numpy as np
    
    # Compute pseudoinverse
    J_pinv = np.linalg.pinv(J)
    
    # Compute null space projector
    null_projector = np.eye(J.shape[1]) - np.dot(J_pinv, J)
    
    # Compute joint velocities
    q_dot = np.dot(J_pinv, x_dot) + np.dot(null_projector, q_dot_0)
    
    return q_dot
```

## Workspace Analysis

### Types of Workspace
- **Reachable Workspace**: All points reachable by the end-effector
- **Dexterous Workspace**: Points reachable with any orientation
- **Constant-Orientation Workspace**: Points reachable with specific orientation
- **Task-Oriented Workspace**: For specific tasks or applications

### Workspace Determination Methods
- **Geometric Approach**: Using joint limits and link lengths
- **Numerical Sampling**: Evaluating random configurations
- **Boundary Tracing**: Following workspace boundaries
- **Analytical Methods**: For simple manipulators
- **Visualization Techniques**: Representing 3D workspaces

### Workspace Optimization
- **Task Placement**: Positioning tasks within workspace
- **Robot Placement**: Positioning robot for optimal task execution
- **Design Optimization**: Modifying link lengths and joint limits
- **Performance Maps**: Visualizing performance metrics over workspace

## Applications of Kinematics

### Robot Programming
- **Joint Space Trajectories**: Planning smooth joint motions
- **Cartesian Trajectories**: Planning end-effector paths
- **Online Path Correction**: Adjusting trajectories during execution
- **Singularity Avoidance**: Planning paths away from singularities

### Calibration
- **Kinematic Calibration**: Identifying actual DH parameters
- **Error Modeling**: Representing manufacturing and assembly errors
- **Calibration Techniques**: Using measurements to update parameters
- **Performance Evaluation**: Measuring accuracy after calibration

### Human-Robot Interaction
- **Teleoperation**: Mapping human motion to robot motion
- **Impedance Control**: Natural response to human interaction
- **Safety Considerations**: Collision detection and avoidance
- **Collaborative Tasks**: Shared workspace and tasks

## Software Tools for Kinematics

### Libraries and Frameworks
- **Robotics Toolbox for MATLAB**: Peter Corke's library
- **KDL (Kinematics and Dynamics Library)**: Part of Orocos
- **MoveIt**: ROS integration for manipulation
- **Drake**: Planning and control framework
- **PyBullet**: Physics simulation with kinematics support

### Example Code: Forward Kinematics with Modern Libraries
```python
def kdl_forward_kinematics_example():
    """
    Example of forward kinematics using PyKDL.
    """
    import PyKDL as kdl
    
    # Create a kinematic chain
    chain = kdl.Chain()
    
    # Add segments with joints and links
    frame = kdl.Frame()
    joint = kdl.Joint(kdl.Joint.RotZ)  # Revolute joint around Z
    segment = kdl.Segment(joint, frame)
    chain.addSegment(segment)
    
    # Add more segments...
    # ...
    
    # Create solver
    fk_solver = kdl.ChainFkSolverPos_recursive(chain)
    
    # Set joint positions
    n_joints = chain.getNrOfJoints()
    q = kdl.JntArray(n_joints)
    q[0] = 0.5  # First joint position
    q[1] = 0.3  # Second joint position
    # ... set other joint positions
    
    # Solve forward kinematics
    end_effector_frame = kdl.Frame()
    result = fk_solver.JntToCart(q, end_effector_frame)
    
    # Extract position and orientation
    position = end_effector_frame.p
    rotation = end_effector_frame.M
    
    return {
        'position': [position[0], position[1], position[2]],
        'rotation': [
            [rotation[0, 0], rotation[0, 1], rotation[0, 2]],
            [rotation[1, 0], rotation[1, 1], rotation[1, 2]],
            [rotation[2, 0], rotation[2, 1], rotation[2, 2]]
        ]
    }
```

## Practical Exercises

### Exercise 1: Forward Kinematics of a 3-DOF Planar Robot
Implement the forward kinematics for a 3-DOF planar robot and visualize its workspace.

### Exercise 2: Inverse Kinematics of a 6-DOF Robot Arm
Develop analytical and numerical inverse kinematics solutions for a 6-DOF robot arm.

### Exercise 3: Jacobian and Manipulability Analysis
Compute the Jacobian matrix for a robot and analyze its manipulability at different configurations.

### Exercise 4: Redundancy Resolution for a 7-DOF Arm
Implement a null space projection method to utilize redundancy for obstacle avoidance.

## Related Topics
- [Force Control](../4.2_Force_Control/README.md): Controlling interaction forces
- [Gripping and Handling](../4.3_Gripping_and_Handling/README.md): End-effector design and control
- [Control Systems](../../02_Control_Systems/README.md): Controller design for manipulators
- [Trajectory Planning](../../05_Locomotion/README.md): Path planning and trajectory generation 
