# PID Controllers in Robotics

## Introduction
Proportional-Integral-Derivative (PID) controllers are the most widely used control algorithms in industry and robotics. Their popularity stems from their simplicity, versatility, and effectiveness in a wide range of applications. This section explores the theory, implementation, and practical applications of PID control in robotic systems.

## Theoretical Foundations

### The PID Equation
The core PID control law is given by:

u(t) = K_p e(t) + K_i ∫e(τ)dτ + K_d de(t)/dt

Where:
- u(t) is the control signal
- e(t) is the error (difference between setpoint and measured value)
- K_p, K_i, K_d are the proportional, integral, and derivative gains
- t is time

### Component Analysis

#### Proportional Term (P)
- **Function**: Provides a control output proportional to the current error
- **Effect**: Reduces rise time; increases overshoot
- **Limitation**: Cannot eliminate steady-state error for most systems

#### Integral Term (I)
- **Function**: Provides a control output based on the accumulated error
- **Effect**: Eliminates steady-state error; can slow down response
- **Limitation**: May cause windup and oscillation

#### Derivative Term (D)
- **Function**: Provides a control output based on the rate of change of error
- **Effect**: Reduces overshoot and settling time; improves stability
- **Limitation**: Amplifies noise; rarely used alone

### Control Loop Architecture
```
          +-------+
Setpoint→ |       |        +------+        +--------+
          | Error | Control|      | Output |        |
       +→ |       |→       | Plant |→      | Sensor |→+
       |  +-------+        +------+        +--------+ |
       |                                              |
       +----------------------------------------------+
                         Feedback
```

## Implementation Techniques

### Discrete Time Implementation
For digital controllers, the PID algorithm is implemented in discrete time:

u[k] = K_p e[k] + K_i ∑(j=0 to k) e[j]Ts + K_d (e[k] - e[k-1])/Ts

Where:
- Ts is the sampling time
- k is the current time step

### Anti-Windup Strategies
1. **Clamping**: Limiting the integral term when the controller saturates
2. **Back-calculation**: Reducing the integral term when saturation occurs
3. **Conditional integration**: Only integrate when certain conditions are met

### Derivative Filtering
To reduce noise sensitivity, the derivative term is often filtered:

D_filtered = D / (1 + N·Ts)

Where N is the filter coefficient (typically 8-20).

### Auto-Tuning Methods
1. **Ziegler-Nichols**: Classic method based on ultimate gain and period
2. **Cohen-Coon**: Better for processes with significant dead time
3. **Relay Feedback**: Automated method using limit cycles
4. **ITAE Optimization**: Minimizes integral time-weighted absolute error

## Practical Applications in Robotics

### Joint Position Control
- **Application**: Controlling the angular position of robot joints
- **Challenges**: Inertia, friction, backlash, coupling effects
- **Implementation**: Cascaded position-velocity loops

### Motor Velocity Control
- **Application**: Controlling the speed of actuators
- **Challenges**: Non-linear motor dynamics, load variations
- **Implementation**: Current/torque inner loop with velocity outer loop

### Mobile Robot Navigation
- **Application**: Path following and position control
- **Challenges**: Nonholonomic constraints, wheel slip
- **Implementation**: Separate controllers for linear and angular velocity

### Drone Stability
- **Application**: Attitude (roll, pitch, yaw) control
- **Challenges**: Fast dynamics, coupling, disturbances
- **Implementation**: Nested loops with inner rate and outer angle controllers

## Software Implementation

### Pseudocode
```python
class PIDController:
    def __init__(self, kp, ki, kd, output_limits=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits
        
        self.previous_error = 0
        self.integral = 0
        self.last_time = None
        
    def compute(self, setpoint, measured_value, current_time):
        # Calculate error
        error = setpoint - measured_value
        
        # Time difference
        if self.last_time is None:
            dt = 0.1  # Default value on first call
        else:
            dt = current_time - self.last_time
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term with anti-windup
        self.integral += error * dt
        i_term = self.ki * self.integral
        
        # Derivative term with filtering
        derivative = (error - self.previous_error) / dt if dt > 0 else 0
        d_term = self.kd * derivative
        
        # Calculate output
        output = p_term + i_term + d_term
        
        # Apply limits if specified
        if self.output_limits:
            output = max(min(output, self.output_limits[1]), self.output_limits[0])
        
        # Save state for next iteration
        self.previous_error = error
        self.last_time = current_time
        
        return output
```

### Implementation in ROS
```python
#!/usr/bin/env python
import rospy
from std_msgs.msg import Float64

class ROSPIDController:
    def __init__(self):
        rospy.init_node('pid_controller')
        
        # Get parameters
        self.kp = rospy.get_param('~kp', 1.0)
        self.ki = rospy.get_param('~ki', 0.1)
        self.kd = rospy.get_param('~kd', 0.01)
        
        # Create PID controller
        self.pid = PIDController(self.kp, self.ki, self.kd)
        
        # Publishers and subscribers
        self.setpoint_sub = rospy.Subscriber('setpoint', Float64, self.setpoint_callback)
        self.state_sub = rospy.Subscriber('state', Float64, self.state_callback)
        self.control_pub = rospy.Publisher('control', Float64, queue_size=10)
        
        # Initialize variables
        self.setpoint = 0.0
        self.current_state = 0.0
        
        # Timer for control loop
        self.timer = rospy.Timer(rospy.Duration(0.01), self.control_callback)
    
    def setpoint_callback(self, msg):
        self.setpoint = msg.data
    
    def state_callback(self, msg):
        self.current_state = msg.data
    
    def control_callback(self, event):
        # Calculate control output
        current_time = rospy.get_time()
        control = self.pid.compute(self.setpoint, self.current_state, current_time)
        
        # Publish control output
        control_msg = Float64(data=control)
        self.control_pub.publish(control_msg)

if __name__ == '__main__':
    controller = ROSPIDController()
    rospy.spin()
```

## Tuning Guidelines

### Manual Tuning Process
1. Set all gains to zero
2. Increase Kp until oscillations occur
3. Increase Kd to reduce overshoot
4. Increase Ki to eliminate steady-state error
5. Fine-tune all parameters

### Effect of Parameter Changes

| Parameter | Rise Time | Overshoot | Settling Time | Steady-State Error | Stability |
|-----------|-----------|-----------|---------------|-------------------|-----------|
| Kp ↑      | Decrease  | Increase  | Small change  | Decrease          | Degrade   |
| Ki ↑      | Decrease  | Increase  | Increase      | Eliminate         | Degrade   |
| Kd ↑      | Small effect | Decrease | Decrease     | No effect         | Improve if small |

### Typical Values for Different Systems
- **Position control**: Kp=1-10, Ki=0.1-1.0, Kd=0.1-0.5
- **Velocity control**: Kp=0.5-1.0, Ki=0.1-0.5, Kd=0-0.1
- **Temperature control**: Kp=1-100, Ki=0.01-1.0, Kd=0-0.1

## Common Problems and Solutions

### Oscillations
- **Cause**: High proportional gain or integral windup
- **Solution**: Reduce Kp, add derivative action, implement anti-windup

### Slow Response
- **Cause**: Low proportional gain, high integral time
- **Solution**: Increase Kp, decrease Ti (increase Ki)

### Steady-State Error
- **Cause**: Lack of integral action, system disturbances
- **Solution**: Add or increase integral gain

### Noise Sensitivity
- **Cause**: High derivative gain amplifying sensor noise
- **Solution**: Filter the derivative term, reduce Kd

## Advanced PID Variations

### Cascaded PID Control
Using nested PID loops where the output of one controller becomes the setpoint for another.

### Feed-Forward Compensation
Adding a model-based term to the control signal to compensate for known disturbances.

### Gain Scheduling
Changing PID parameters based on operating conditions or system states.

### Model-Based PID
Incorporating system models to enhance controller performance.

## Assessment and Exercises

### Knowledge Check
1. What causes integral windup and how can it be prevented?
2. Why might derivative control be omitted in some applications?
3. How does sampling time affect PID controller performance?
4. What is the difference between parallel and series PID forms?

### Practical Exercises
1. **Basic Implementation**: Create a PID controller for a simulated first-order system
2. **Motor Control**: Implement velocity control for a DC motor
3. **Line Following**: Design a PID controller for a line-following robot
4. **Balance Control**: Create a controller for a simulated inverted pendulum

## Related Topics
- [State-Space Control](../2.2_StateSpace_Control/README.md): Modern control techniques
- [Kalman Filters](../2.3_Kalman_Filters/README.md): State estimation for control
- [Model Predictive Control](../2.4_Model_Predictive_Control/README.md): Advanced predictive control 