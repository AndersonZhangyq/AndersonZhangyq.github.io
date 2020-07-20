## EEG & Video

### 2019 Detecting Events In Video Sequence Of Video-EEG Monitoring

#### Goal

An algorithm for automated detecting diagnostic events in video channel of video and electroencephalographic (EEG) monitoring data

#### Method

1. 计算光流
2. 计算$J(i)=\sum_y\sum_x\sqrt{V_x^2(x,y,i)+V_y^2(x,y,i)}+\delta(i)$
3. 平滑后得$J(I)$
4. 判断是否发出警报

### 2012 Automatic Segmentation of Episodes Containing Epileptic Clonic Seizures in Video Sequences & 2017 Automated video-based detection of nocturnal convulsive seizures in a residential care setting

#### Goal

A method by which a subset of motor seizures can be detected using only remote sensing devices (i.e., not in contact with the subject) such as video cameras.

#### Method

1. 计算光流
2. 平滑
3. Extracting only rates of global motion parameters
4. Gabor Wavelet Technique
5. 阈值

### 2018 Convolutional neural networks for real-time epileptic seizure detection

#### Method

浅层CNN，softmax二分类，红外+深度作为输入

### 2019 Motion Signatures for the Analysis of Seizure Evolution in Epilepsy

#### Method

1. Face Detection+Hand Detection
2. Pose Estimation and get hand position
3. Optical flow
4. Motion signature (Sum optical flow along axis X, stack by time)

### 2019 Epileptic seizure classification using the NeuroMov database

### Method

Detect bed + action recognition (CNN for feature extraction + MLP)

### 2019 Aberrant epileptic seizure identification: A computer vision perspective

#### Method

1. Conv-LSTM to extract spatiotemporal representations
2. **Motion capture libraries** from 119 seizures of 28 patients
3. Cosine similarity distance between a test representation and the libraries from five aberrant seizures separate to the main dataset, to identify test seizures with unusual patterns that do not conform to known behavior.

### 2018 Vision based Methodology for Diagnosis of Convulsion Patients (Optical Flow)

#### Method

Optical Flow with smoothness (Eq 4)

### 2019 Machine learning applications in epilepsy

小综述，提到了2012年以前的几个方法

- [ ] ### 2019 Neonatal Seizures—Are We there Yet? (Further Dig)

### 2018 Spatial Temporal GRU Convnets For Vision-based Real Time Epileptic Seizure Detection

#### Goal

A novel marker-free visionbased monitoring method

#### Method

Optical Flow $\rarr$  Temporal ConvNet $\rarr$ Consensus $\rarr$ $feature_1$

Frame $\rarr$ Spatial ConvNet $\rarr$   $feature_2$$\rarr$ GRU $\rarr$ 对早期的错误判断惩罚更低使得能较早检测出惊厥 $\rarr$ result

Concat($feature_1$,$feature_2$) + result $\rarr$ refine

### 2018 Deep Motion Analysis for Epileptic Seizure Classification

#### Method

Face Detection $\rarr$ Feature

Pose $\rarr$ Refined (If the **distance** between the estimated points is **within a certain range** when compared to the previous position, the point is labelled as **valid**. If the point is **rejected**, the algorithm searches for the nearest neighbor inside the area that fulfils the criteria and continues its tracking to the next frame)

Temporal information(LSTM, use the extracted spatial feature)

Fuse

### 2018 Detection of Infantile Movement Disorders in Video Data Using Deformable Part-Based Model

#### Method

Motion analysis: Calculate angle

### 2012 Low-Complexity Image Processing for Real-Time Detection of Neonatal Clonic Seizures 

#### Goal

An innovative low complexity image-processing-based approach to the detection of clonic neonatal seizures.

#### Method

Diff of frame $\rarr$ 二值化（压缩信息，使用阈值，但是这个阈值时统计归纳的，可能是数据相关度较高的超参） $\rarr$ m领域内侵蚀 $\rarr$ 累加得到运动信息 $\rarr$ 周期性检测（惊厥行为是重复的，会呈现周期性，所以运动特征也应该是周期性的） 

### 2015 A Neural Network Based Infant Monitoring System to Facilitate Diagnosis of Epileptic Seizures

#### Goal

An epileptic seizure detection system that can choose/combine evaluation indices calculated from video images and EEG signals.

#### Method

Diff of frame $\rarr$ 二值化 $\rarr$ 四象限划分（如果婴儿不在图像中间这个划分可能会失效）加全图共五个区域 $\rarr$ 每个区域的像素值累加起来就是运动特征 $\rarr$ $G_i^k=\frac{1}{w}\sum_{i=l-w+1}^l M_i^k$，$M_{ik}^{diff}=G_i^k+\min{(G_{i-v+1}^k,\cdots,G_i^k)}$，每个区域计算好后取均值 $\rarr$ concat EEG指标后输入LLGMM

### 2017 Monitoring infants by automatic video processing: A unified approach to motion analysis

#### Goal

A unified approach to contact-less and low-cost video processing for automatic detection of neonatal diseases characterized by specific movement patterns

#### Method

预处理输入 $\rarr$ 灰度化、Finite Impulse Response on Diff of frame $\rarr$ 二值化