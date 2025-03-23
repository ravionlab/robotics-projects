# Computer Vision in Robotics

## Introduction
Computer Vision enables robots to interpret and understand visual information from the world, a critical capability for intelligent systems. This section explores fundamental concepts, algorithms, and applications of computer vision in robotics, ranging from basic image processing to advanced deep learning techniques for scene understanding and object manipulation.

## Image Formation and Representation

### Camera Models
- **Pinhole Camera Model**: Basic projective geometry
- **Camera Intrinsics**: Focal length, principal point, lens distortion
- **Camera Extrinsics**: Position and orientation in world coordinates
- **Stereo Camera Geometry**: Epipolar constraints, disparity, triangulation
- **Omnidirectional Cameras**: Wide field of view, catadioptric systems

### Image Representation
- **Color Spaces**: RGB, HSV, YUV, Lab
- **Image Pyramids**: Multi-scale representation
- **Feature Spaces**: Gradient, texture, frequency domain
- **3D Representations**: Point clouds, depth maps, voxel grids
- **Semantic Representations**: Labeled regions, object instances

## Image Processing Fundamentals

### Filtering and Enhancement
- **Convolution**: Kernel-based operations
- **Linear Filters**: Gaussian, Laplacian, Sobel
- **Nonlinear Filters**: Median, bilateral, anisotropic diffusion
- **Morphological Operations**: Erosion, dilation, opening, closing
- **Frequency Domain**: Fourier transform, filtering in frequency space

```python
def apply_gaussian_filter(image, kernel_size=5, sigma=1.0):
    """Apply Gaussian filter to an image."""
    import cv2
    import numpy as np
    
    # Create Gaussian kernel
    kernel = cv2.getGaussianKernel(kernel_size, sigma)
    kernel_2d = np.outer(kernel, kernel.transpose())
    
    # Apply filter using convolution
    filtered_image = cv2.filter2D(image, -1, kernel_2d)
    
    return filtered_image
```

### Edge and Feature Detection
- **Edge Operators**: Sobel, Canny, Laplacian of Gaussian
- **Corner Detection**: Harris, Shi-Tomasi, FAST
- **Blob Detection**: Difference of Gaussians, Hough circles
- **Feature Descriptors**: SIFT, SURF, ORB, BRIEF
- **Feature Matching**: Nearest neighbor, ratio test, RANSAC

```python
def detect_and_match_features(img1, img2):
    """Detect and match ORB features between two images."""
    import cv2
    
    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=1000)
    
    # Find keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    # Match descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    return kp1, kp2, matches
```

### Segmentation
- **Thresholding**: Binary, adaptive, Otsu's method
- **Region-Based**: Region growing, watershed
- **Clustering**: K-means, mean shift
- **Graph-Based**: Graph cuts, normalized cuts
- **Model-Based**: Active contours, level sets

## 3D Vision

### Stereo Vision
- **Stereo Matching**: Finding correspondences between image pairs
- **Disparity Computation**: Block matching, semi-global matching
- **Depth Reconstruction**: Triangulation, baseline considerations
- **Epipolar Geometry**: Fundamental matrix, essential matrix
- **Rectification**: Aligning image pairs for efficient matching

### Structure from Motion
- **Feature Tracking**: Over multiple frames
- **Camera Pose Estimation**: Perspective-n-Point (PnP), essential matrix decomposition
- **Bundle Adjustment**: Jointly optimizing camera poses and 3D points
- **Dense Reconstruction**: Multi-view stereo, photometric consistency
- **Applications**: Visual odometry, 3D model building

### RGB-D Processing
- **Depth Sensing Technologies**: Structured light, time-of-flight, stereo
- **Point Cloud Generation**: Converting depth images to 3D points
- **Surface Reconstruction**: Poisson surface reconstruction, TSDF
- **Registration**: ICP algorithm, point-to-plane ICP
- **Scene Flow**: 3D motion estimation

```python
def depth_to_point_cloud(depth_image, camera_intrinsics):
    """Convert depth image to point cloud."""
    import numpy as np
    
    # Get image dimensions
    height, width = depth_image.shape
    
    # Create pixel coordinate grid
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    
    # Unpack camera intrinsics
    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
    
    # Convert to normalized coordinates
    x = (u - cx) * depth_image / fx
    y = (v - cy) * depth_image / fy
    z = depth_image
    
    # Stack to create point cloud
    points = np.stack((x, y, z), axis=2)
    
    # Reshape to list of points
    points = points.reshape(-1, 3)
    
    # Filter out invalid points (zero depth)
    valid_points = points[z.flatten() > 0]
    
    return valid_points
```

## Object Recognition and Detection

### Traditional Approaches
- **Template Matching**: Direct comparison with templates
- **Histogram of Oriented Gradients (HOG)**: Gradient-based features
- **Bag of Visual Words**: Feature quantization and histograms
- **Deformable Part Models**: Modeling object parts and relations
- **Support Vector Machines**: Classical approach to classification

### Convolutional Neural Networks
- **Architecture Components**: Convolutional layers, pooling, normalization
- **Region-Based CNNs**: R-CNN, Fast R-CNN, Faster R-CNN
- **Single-Shot Detectors**: YOLO, SSD
- **Feature Pyramids**: FPN, RetinaNet
- **Backbone Networks**: ResNet, MobileNet, EfficientNet

```python
def load_and_predict_with_yolo(image_path, model_path):
    """Object detection using YOLOv5 model."""
    import torch
    
    # Load model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
    
    # Inference
    results = model(image_path)
    
    # Process results
    detections = results.pandas().xyxy[0]  # Bounding box coordinates, confidence, class
    
    return detections
```

### Instance Segmentation
- **Mask R-CNN**: Region-based instance segmentation
- **YOLACT**: Real-time instance segmentation
- **PointRend**: Point-based rendering for detailed masks
- **Panoptic Segmentation**: Combining semantic and instance segmentation
- **Applications**: Object manipulation, scene understanding

### Object Pose Estimation
- **PnP-Based Methods**: 2D-3D correspondences
- **Learning-Based**: Direct regression, keypoint detection
- **Dense Methods**: Iterative closest point, dense correspondence
- **6D Pose**: Position and orientation estimation
- **Applications**: Grasping, object manipulation

## Semantic Understanding

### Semantic Segmentation
- **Fully Convolutional Networks**: End-to-end pixel classification
- **Encoder-Decoder Architectures**: U-Net, SegNet
- **Dilated Convolutions**: Expanding receptive field
- **Transformers for Segmentation**: SETR, Segformer
- **Real-time Methods**: EfficientPS, SwiftNet

### Scene Parsing
- **Scene Graphs**: Objects and their relationships
- **Context Modeling**: Incorporating spatial and semantic context
- **Hierarchical Representations**: Multi-scale scene understanding
- **Affordance Detection**: Understanding possible actions on objects
- **Applications**: Task planning, human-robot interaction

### Visual Recognition Systems
- **Visual Question Answering**: Understanding and answering questions about images
- **Image Captioning**: Generating natural language descriptions
- **Visual Grounding**: Localizing objects based on language descriptions
- **Multimodal Learning**: Combining vision with other modalities
- **Knowledge Incorporation**: Using prior knowledge for better understanding

## Motion and Tracking

### Optical Flow
- **Lucas-Kanade Method**: Sparse feature tracking
- **Horn-Schunck Method**: Dense flow estimation
- **FlowNet, PWC-Net**: Deep learning approaches
- **Applications**: Motion segmentation, video analysis
- **Challenges**: Occlusions, lighting changes, large displacements

```python
def compute_optical_flow(prev_frame, curr_frame):
    """Compute optical flow using Lucas-Kanade method."""
    import cv2
    import numpy as np
    
    # Convert to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    
    # Find good features to track
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, 
                                       qualityLevel=0.3, minDistance=7)
    
    # Calculate optical flow
    curr_pts, status, error = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, prev_pts, None)
    
    # Select good points
    good_prev = prev_pts[status == 1]
    good_curr = curr_pts[status == 1]
    
    return good_prev, good_curr
```

### Visual Object Tracking
- **Correlation Filters**: MOSSE, KCF, DSST
- **Siamese Networks**: SiamFC, SiamRPN
- **Online Learning**: MDNet, UpdateNet
- **Transformer-based**: TransT, STARK
- **Long-term Tracking**: Re-detection strategies, global search

### Multiple Object Tracking
- **Tracking-by-Detection**: Detecting then associating
- **Data Association**: Hungarian algorithm, JPDA
- **Identity Management**: Re-identification, appearance modeling
- **DeepSORT**: Combining CNN features with Kalman filtering
- **Graph Neural Networks**: Modeling object interactions

## Applications in Robotics

### Visual Servoing
- **Image-Based VS**: Controlling robot to achieve desired image features
- **Position-Based VS**: Using visual information to estimate pose
- **Hybrid Approaches**: Combining image and position-based control
- **Learning-Based VS**: End-to-end learning of control policies
- **Applications**: Precision manipulation, docking, navigation

### Visual Navigation
- **Visual Teach and Repeat**: Following previously demonstrated paths
- **Visual Homing**: Reaching target positions using visual cues
- **Topological Navigation**: Graph-based navigation using visual landmarks
- **End-to-End Learning**: Directly mapping images to navigation commands
- **Vision-Language Navigation**: Following natural language instructions

### Manipulation and Grasping
- **Grasp Detection**: Finding stable grasp poses
- **Visual Tactile Integration**: Combining vision with touch
- **Learning from Demonstration**: Imitating human demonstrations
- **Sim-to-Real Transfer**: Training in simulation, deploying in real world
- **Challenges**: Cluttered environments, novel objects, dynamics

### Autonomous Driving
- **Road Detection**: Lane finding, drivable area segmentation
- **Traffic Participant Detection**: Vehicles, pedestrians, cyclists
- **Behavior Prediction**: Forecasting trajectories and intentions
- **Scene Understanding**: Traffic signs, signals, infrastructure
- **End-to-End Approaches**: Direct mapping from camera to control

## Deep Learning for Vision

### Network Architectures
- **CNN Backbones**: ResNet, EfficientNet, ConvNeXt
- **Attention Mechanisms**: Self-attention, cross-attention
- **Vision Transformers**: ViT, Swin Transformer, DeiT
- **Multi-Modal Networks**: CLIP, DALL-E, ImageBERT
- **Efficient Architectures**: MobileNet, EfficientDet, YOLOv5

### Training Strategies
- **Transfer Learning**: Fine-tuning pre-trained models
- **Self-Supervised Learning**: Learning without labels
- **Few-Shot Learning**: Generalizing from few examples
- **Domain Adaptation**: Adapting to new environments
- **Curriculum Learning**: Structured training progression

### Deployment Considerations
- **Model Compression**: Quantization, pruning, distillation
- **Hardware Acceleration**: GPU, FPGA, TPU, edge devices
- **Latency-Accuracy Tradeoffs**: Real-time performance balance
- **Robustness**: Handling domain shift, adversarial examples
- **Uncertainty Estimation**: Knowing when the model is uncertain

## Practical Implementation

### OpenCV in Robotics
- **Image Acquisition**: Camera interfacing, calibration
- **Image Processing Pipeline**: Preprocessing, feature extraction
- **Tracking and Detection**: Implementation of algorithms
- **3D Vision**: Stereo matching, depth processing
- **Integration with ROS**: Image transport, visualization

### Deep Learning Frameworks
- **PyTorch**: Dynamic computation graphs
- **TensorFlow/Keras**: Static graphs, deployment options
- **ONNX**: Framework interoperability
- **Model Optimization**: TensorRT, OpenVINO
- **Edge Deployment**: TFLite, PyTorch Mobile

### ROS Integration
```python
#!/usr/bin/env python
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D

class ObjectDetectionNode:
    def __init__(self):
        rospy.init_node('object_detection_node')
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Load model (e.g., YOLO, SSD)
        self.model = self.load_model()
        
        # Publishers and subscribers
        self.image_sub = rospy.Subscriber('camera/image_raw', Image, self.image_callback)
        self.detection_pub = rospy.Publisher('detections', Detection2DArray, queue_size=10)
        self.vis_pub = rospy.Publisher('detection_visualization', Image, queue_size=10)
        
    def load_model(self):
        # Example function to load a deep learning model
        # In a real implementation, use appropriate framework (PyTorch, TF, etc.)
        net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        return net
    
    def image_callback(self, msg):
        # Convert ROS Image to OpenCV format
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # Run detection
        detections = self.detect_objects(cv_image)
        
        # Create detection message
        detection_msg = Detection2DArray()
        detection_msg.header = msg.header
        
        # Process detections
        vis_image = cv_image.copy()
        for det in detections:
            x, y, w, h, conf, class_id = det
            
            # Create ROS detection message
            d = Detection2D()
            d.header = msg.header
            d.bbox.center.x = x + w/2
            d.bbox.center.y = y + h/2
            d.bbox.size_x = w
            d.bbox.size_y = h
            
            # Add to array
            detection_msg.detections.append(d)
            
            # Draw on visualization image
            cv2.rectangle(vis_image, (int(x), int(y)), 
                          (int(x+w), int(y+h)), (0, 255, 0), 2)
        
        # Publish detection results
        self.detection_pub.publish(detection_msg)
        
        # Publish visualization
        vis_msg = self.bridge.cv2_to_imgmsg(vis_image, "bgr8")
        self.vis_pub.publish(vis_msg)
    
    def detect_objects(self, image):
        # Preprocess image
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), 
                                    swapRB=True, crop=False)
        self.model.setInput(blob)
        
        # Forward pass
        layer_names = self.model.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in self.model.getUnconnectedOutLayers()]
        outputs = self.model.forward(output_layers)
        
        # Process outputs
        detections = []
        # ... process network outputs to get bounding boxes ...
        # (Implementation details omitted for brevity)
        
        return detections

if __name__ == '__main__':
    try:
        node = ObjectDetectionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
```

## Challenges and Future Directions

### Robustness to Real-World Conditions
- **Illumination Variation**: Handling different lighting conditions
- **Weather Effects**: Rain, snow, fog, glare
- **Motion Blur**: Fast robot or object movement
- **Domain Gaps**: Difference between training and deployment environments
- **Adversarial Robustness**: Resilience to attacks or manipulation

### Lifelong Learning and Adaptation
- **Continual Learning**: Updating models without forgetting
- **Active Learning**: Selecting informative samples for labeling
- **Few-Shot Learning**: Quickly adapting to new objects or scenes
- **Self-Supervised Adaptation**: Adapting without human supervision
- **Knowledge Distillation**: Transferring knowledge between models

### Explainable Computer Vision
- **Visualization Techniques**: Grad-CAM, saliency maps
- **Interpretable Models**: Decision trees, rule-based systems
- **Attribution Methods**: Understanding model decisions
- **Human-in-the-Loop**: Incorporating human feedback
- **Trust and Verification**: Ensuring reliable vision systems

## Learning Resources

### Fundamental Textbooks
- "Computer Vision: Algorithms and Applications" by Richard Szeliski
- "Multiple View Geometry in Computer Vision" by Hartley and Zisserman
- "Deep Learning" by Goodfellow, Bengio, and Courville
- "Deep Learning for Vision Systems" by Mohamed Elgendy
- "Probabilistic Robotics" by Thrun, Burgard, and Fox

### Online Courses
- Stanford CS231n: Convolutional Neural Networks for Visual Recognition
- University of Michigan: Deep Learning for Computer Vision
- Coursera: Computer Vision Specialization
- Udacity: Computer Vision Nanodegree
- EdX: Robotics: Vision Intelligence and Machine Learning

### Open-Source Projects
- OpenCV: Computer vision library
- YOLO: You Only Look Once object detection
- Detectron2: Facebook AI Research's detection framework
- TorchVision: PyTorch's computer vision library
- MMDetection: Open MMLab detection toolbox

## Practical Exercises

### Exercise 1: Camera Calibration
Calibrate a camera and use the intrinsic parameters to undistort images and calculate 3D positions from pixel coordinates.

### Exercise 2: Feature-Based Object Recognition
Implement a system to recognize objects using feature extraction, description, and matching techniques.

### Exercise 3: Deep Learning for Object Detection
Train and deploy a neural network for detecting and localizing objects relevant to a robotic task.

### Exercise 4: Visual Servoing
Develop a control system that uses visual feedback to guide a robot arm to a target position.

## Related Topics
- [Sensor Fusion](../3.1_Sensor_Fusion/README.md): Combining vision with other sensors
- [SLAM](../3.2_SLAM/README.md): Using visual data for mapping and localization
- [Manipulation](../../04_Manipulation/README.md): Visual servoing for robot manipulation
- [Locomotion](../../05_Locomotion/README.md): Vision for navigation and obstacle avoidance 
