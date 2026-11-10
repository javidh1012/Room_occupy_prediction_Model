# 🏢 Room Occupancy Prediction

An end-to-end Machine Learning web application designed to estimate and predict room occupancy count in real time based on multi-sensor environmental data including temperature, light intensity, sound levels, carbon dioxide ($CO_2$) concentration, and PIR motion detection.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://room-occupy-prediction-model.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![PyCaret](https://img.shields.io/badge/PyCaret-3.3.2-blueviolet)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4.2-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 🔗 Live Demo

Experience the deployed application here:  
👉 **[https://room-occupy-prediction-model.streamlit.app/](https://room-occupy-prediction-model.streamlit.app/)**

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Live Demo](#-live-demo)
- [Input Features & Sensor Attributes](#-input-features--sensor-attributes)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Local Installation & Setup](#-local-installation--setup)
- [Usage Guide](#-usage-guide)
- [Model Details](#-model-details)
- [License](#-license)

---

## 📖 Overview

Accurate occupancy estimation in indoor environments plays a critical role in **smart building automation**, **HVAC (heating, ventilation, and air conditioning) optimization**, **energy conservation**, and **space management**. 

Traditional methods relying on surveillance cameras raise significant privacy concerns and require high computational overhead. The **Room Occupancy Prediction** system solves this by using non-intrusive, privacy-preserving IoT sensor streams:
- **Temperature & Light**: Detect ambient thermal footprint and illumination changes caused by occupants.
- **Sound Levels**: Measure acoustic activity inside the room.
- **$CO_2$ & Slope**: Measure metabolic respiration rates and changes in carbon dioxide buildup over time.
- **PIR Sensors**: Detect physical motion in multiple zones of the room.

The pre-trained Machine Learning pipeline evaluates these multi-modal sensor readings to accurately infer the number of occupants present in the room.

---

## ✨ Key Features

- **Privacy-Preserving**: Accurately estimates occupancy without cameras, microphones recording audio content, or intrusive monitoring.
- **Interactive Web Interface**: Streamlit UI organized in a clean 4-column layout for rapid input of all 16 sensor readings.
- **Pre-populated Baselines**: Input fields are pre-loaded with dataset mean values for immediate testing and quick experimentation.
- **Real-Time Prediction**: Instant estimation of the room occupancy count with a single click.
- **Automated Pipeline Handling**: Robust model loading with caching and feature alignment via PyCaret and Scikit-Learn.

---

## 📊 Input Features & Sensor Attributes

The model processes **16 continuous environmental sensor variables**:

### 1. Temperature Sensors (°C)
| Feature | Type | Description | Default Mean |
| :--- | :--- | :--- | :--- |
| `S1_Temp` | Numeric | Sensor 1 temperature reading | `25.4558 °C` |
| `S2_Temp` | Numeric | Sensor 2 temperature reading | `25.5490 °C` |
| `S3_Temp` | Numeric | Sensor 3 temperature reading | `25.0588 °C` |
| `S4_Temp` | Numeric | Sensor 4 temperature reading | `25.7565 °C` |

### 2. Light Sensors (Lux)
| Feature | Type | Description | Default Mean |
| :--- | :--- | :--- | :--- |
| `S1_Light` | Numeric | Sensor 1 illuminance level | `25.5531 Lux` |
| `S2_Light` | Numeric | Sensor 2 illuminance level | `26.1282 Lux` |
| `S3_Light` | Numeric | Sensor 3 illuminance level | `34.5756 Lux` |
| `S4_Light` | Numeric | Sensor 4 illuminance level | `13.1883 Lux` |

### 3. Sound Sensors (Volts / Sound Activity)
| Feature | Type | Description | Default Mean |
| :--- | :--- | :--- | :--- |
| `S1_Sound` | Numeric | Sensor 1 acoustic signal reading | `0.1678` |
| `S2_Sound` | Numeric | Sensor 2 acoustic signal reading | `0.1194` |
| `S3_Sound` | Numeric | Sensor 3 acoustic signal reading | `0.1604` |
| `S4_Sound` | Numeric | Sensor 4 acoustic signal reading | `0.1045` |

### 4. Carbon Dioxide & Motion Sensors
| Feature | Type | Description | Default Mean |
| :--- | :--- | :--- | :--- |
| `S5_CO2` | Numeric | Carbon dioxide concentration in parts per million (ppm) | `462.2035 ppm` |
| `S5_CO2_Slope` | Numeric | Rate of change / slope of $CO_2$ concentration over time | `-0.0141` |
| `S6_PIR` | Numeric | Passive Infrared motion detector 1 (activity indicator) | `0.0887` |
| `S7_PIR` | Numeric | Passive Infrared motion detector 2 (activity indicator) | `0.0792` |

---

## 🛠️ Tech Stack

- **Application Framework**: [Streamlit](https://streamlit.io/)
- **Machine Learning & AutoML**: [PyCaret](https://pycaret.org/) & [Scikit-Learn](https://scikit-learn.org/) (v1.4.2)
- **Data Manipulation**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Model Serialization**: [Joblib](https://joblib.readthedocs.io/)
- **Hosting & Deployment**: [Streamlit Community Cloud](https://streamlit.io/cloud)

---

## 📁 Project Structure

```plaintext
Room_occupy_prediction/
│
├── Room_Occupancy_prediction.joblib   # Trained Machine Learning pipeline / model
├── main.py                            # Streamlit web application & prediction interface
├── requirements.txt                   # Project Python dependencies
├── logs.log                           # Execution and training/runtime logs
└── README.md                          # Documentation and setup instructions
```

---

## 💻 Local Installation & Setup

To run this application locally on your machine, follow these steps:

### 1. Prerequisites
- **Python 3.10 or 3.11** is recommended for compatibility with PyCaret 3.3.x and Scikit-Learn 1.4.2.

### 2. Clone or Navigate to the Repository
```bash
git clone https://github.com/<your-username>/Room_occupy_prediction.git
cd Room_occupy_prediction
```

### 3. Create and Activate a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

> **Note on Scikit-Learn version:** The serialized model requires `scikit-learn==1.4.2` as specified in `requirements.txt`. Ensure this exact version is installed to prevent unpickling mismatches.

### 5. Run the Streamlit Application
```bash
streamlit run main.py
```

The app will start and be accessible at `http://localhost:8501`.

---

## 🚀 Usage Guide

1. Open the [Live Demo](https://room-occupy-prediction-model.streamlit.app/) or run locally.
2. Review or adjust the sensor readings in the 4-column form:
   - Modify **Temperature** readings (`S1_Temp` – `S4_Temp`).
   - Adjust **Light** levels (`S1_Light` – `S4_Light`).
   - Set **Sound** values (`S1_Sound` – `S4_Sound`).
   - Set **$CO_2$** levels, slope, and **PIR** motion activity flags.
3. Click the **"Predict occupancy"** button.
4. The estimated **Predicted room occupancy** count is displayed as a clean metric card on the dashboard.

---

## 🧠 Model Details

- **Target Variable**: `Room_Occupancy_Count` (Estimated number of people in the room, e.g., 0, 1, 2, 3...)
- **Serialized Artifact**: `Room_Occupancy_prediction.joblib`
- **Pipeline Architecture**: Includes automated preprocessing and feature alignment handled by PyCaret/Scikit-Learn.
- **Inference Flow**: Incoming readings from the Streamlit UI are formatted into a DataFrame matching the model's expected feature schema (including sensor readings and timestamp column mappings) before computing the predicted room occupancy count.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE). Feel free to use, modify, and distribute it for academic or commercial purposes.

