# Vision-based Anomaly Detection to Optimize Logistic Processes

## Project Summary

### Introduction
This project focuses on detecting unknown objects in live videos at WasteAnt plants to identify potential hazards. The aim is to use computer vision and machine learning techniques to detect anomalies in waste processing to enhance the safety and efficiency of waste management.

### Dataset
- **Source**: WasteAnt datasets
- **Data Type**: Videos of waste falling into WasteAnt plants
- **Focus**: Analyzing object velocities (optical flow) and detecting patterns via the LSTM (Long Short-Term Memory) classifier.

### Exploratory Data Analysis
- **Optical Flow**: Used to estimate the motion of objects between consecutive video frames.
- **Visualization**: Techniques such as HSV (Hue Saturation Value) and drawing arrows were explored to visualize optical flow and detect motion patterns.

### Challenges
- **Cluttered Environment**: The highly cluttered and dynamic environment made object detection difficult due to occlusions, motion blur, and image distortions.
- **Illumination**: Variations in illumination affected anomaly detection.
- **Perspective**: Camera angles impacted the detection of anomalies.
- **Sparse or Dense Environment**: The dense environment required a robust model to handle fast-moving and cluttered objects.

### Data Preprocessing
- **Cleaning**: Rescaling images and extracting Region of Interest (ROI).
- **Optical Flow Calculation**: Used the Gunnar Farneback method to estimate motion between frames.
- **Sequence Creation**: Created sequences of past and future velocities for training the LSTM model.
- **Threshold Identification**: Established thresholds for detecting anomalies based on velocity patterns.

### Model Development & Evaluation
- **LSTM Model**: Used for capturing temporal information and detecting anomalies in motion patterns.
- **Training**: Various models and parameters were tested to achieve the best performance.
  - **Basic LSTM Model**: Initial model with single LSTM and Dense layers.
  - **TimeDistributed Layer**: Improved accuracy but not consistent.
  - **Stacked LSTM Layers**: Provided better accuracy and robustness.
- **Model Performance**:
  - **Train Accuracy**: 0.842
  - **Test Accuracy**: 0.636
  - **Dummy Accuracy**: 0.618 and 0.513 for different test videos.

### Final Model Choice
The final model used a stacked LSTM architecture, which provided the best balance of accuracy and robustness for detecting anomalies in waste processing.

### Further Work
1. **Feature Extraction**: Explore collecting patterns of waste features from images and comparing them to detect anomalies.
2. **Enhance Robustness**: Improve the model's ability to handle various types of waste and different environmental conditions.
3. **Integrate Real-Time Deployment**: Develop a system for real-time anomaly detection in WasteAnt plants.
4. **Expand Dataset**: Gather more data to improve model training and testing.

This project aims to develop a robust and efficient anomaly detection system to optimize logistic processes in waste management, ensuring safety and improving operational efficiency.
