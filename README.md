# -AI-Driven-Intrusion-Detection-System-for-IoT-Networks

📌 Overview
This repository contains the official implementation of the research paper: "An Efficient AI-Driven Intrusion Detection System for Resource-Constrained IoT Environments", presented at IEEE IMCET 2026.   The project addresses the critical security challenges in IoT by deploying a lightweight, high-accuracy Intrusion Detection System (IDS) specifically optimized for edge devices with limited computational power.   

🚀 Key FeaturesHigh Precision: 

Achieved 99.25% multiclass accuracy on the comprehensive CICIoT2023 dataset.  

Lightweight Design: Optimized for resource-constrained environments with an inference time of only 0.1386 ms per packet.   

Real-time Integration: Full pipeline integration with Suricata and Telegram Bot API for instantaneous attack notifications.  

Advanced Feature Selection: Utilized Extra Trees and XGBoost for efficient dimensionality reduction without compromising detection rates.   

🛠️ Technical StackLanguages: 

Python (Scikit-learn, TensorFlow Lite, XGBoost).   


Security Tools: Suricata IDS, Network Traffic Analysis.   

Deployment: Edge Computing, Telegram Bot API.   

Algorithms: RNN-LSTM, Random Forest, KNN, and SVM benchmarks. 

📊 Dataset & MethodologyData Source: Used the CICIoT2023 dataset containing over 2.6M samples of modern IoT attacks.   Preprocessing: Feature engineering, missing value imputation, and scaling for optimal ML performance.   
Optimization: Model quantization and pruning to ensure compatibility with IoT hardware.   

📈 Results
Metric,Value
Accuracy,99.25%
Inference Latency,0.1386 ms
Alert Notification Latency,~1.2s (via Telegram)

📖 CitationIf you use this code or research in your work, please cite:Code snippet@inproceedings{hindawi2026efficient,
  title={An Efficient AI-Driven Intrusion Detection System for Resource-Constrained IoT Environments},
  author={Hindawi, Asmaa and El Arid, Amal and others},
  booktitle={5th IEEE International Multidisciplinary Conference on Engineering Technology (IMCET)},
  year={2026},
  location={Beirut, Lebanon}
}

👥 AuthorsAsmaa Hindawi - Lead Researcher & Developer   Dr. Amal El Arid - Research Supervisor   
