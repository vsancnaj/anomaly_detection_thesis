# Anomaly Detection Thesis

## Vision-based anomaly detection in highly cluttered environment to optimize logistic processes
#### Waste-to-Energy plants usually receive high volumes of waste, 15 tons of waste per delivery truck, which makes it very hard to inspect all elements manually. Even for automated systems this case would be very hard because all objects are in a highly cluttered environment, so object detection becomes very difficult. Objects can be occluded by others or fall at very high speeds which can cause motion blur or image distortions. 

#### For this reason, we decided to analyze objects' velocities (optical flow) and analyze its patterns via the LSTM(Long Short Term Memory) classifier. From WasteAnt datasets, we train a temporal LSTM model and then check for anomalies that could be hazardous to the waste management plant. My hypothesis is that anomalies in the way waste falls could be a more robust indicator of an anomaly in the waste batch than a classical object detector.
