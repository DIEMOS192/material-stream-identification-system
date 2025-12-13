# Team Task Division - Material Stream Identification System

## 👤 Member 1: Data Engineer (The Foundation)

- **Goal:** Ensure clean, formatted data is available for the models.
- **Primary Files:** `src/data_loader.py` (New), `src/preprocessing.py` (New)

**Responsibilities:**

1. **Image Loading:** Write functions to read images from `data/raw/` subdirectories.
2. **Preprocessing:** Implement resizing (e.g., to 128x128) and normalization (pixel values 0-1).
3. **Feature Extraction:** Decide on and implement methods to convert images to numbers (e.g., Flattening, Color Histograms, or HOG).
4. **Data Splitting:** Create the train/test split function (80% train, 20% test).

## 👤 Member 2: KNN Specialist

- **Goal:** Build and optimize the K-Nearest Neighbors model.
- **Primary Files:** `src/models/KNN.py`

**Responsibilities:**

1. **Implementation:** Create the `KNNModel` class using `scikit-learn`.
2. **Training:** Write the training logic.
3. **Optimization:** Experiment with different values of `k` (e.g., 3, 5, 7) to find the best accuracy.
4. **Saving:** Ensure the trained model can be saved to a `.pkl` file.

## 👤 Member 3: SVM Specialist

- **Goal:** Build and optimize the Support Vector Machine model.
- **Primary Files:** `src/models/SVM.py`

**Responsibilities:**

1. **Implementation:** Create the `SVMModel` class using `scikit-learn`.
2. **Training:** Write the training logic.
3. **Optimization:** Experiment with different kernels (Linear, RBF, Polynomial) to find the best fit.
4. **Saving:** Ensure the trained model can be saved to a `.pkl` file.

## 👤 Member 4: Application Developer (The Integrator)

- **Goal:** Build the real-time webcam application.
- **Primary Files:** `src/main.py`

**Responsibilities:**

1. **Camera Feed:** Use OpenCV (`cv2`) to capture live video.
2. **Integration:** Load the saved models from Member 2 & 3.
3. **Real-time Prediction:** Pass video frames to the model and get predictions.
4. **UI/Display:** Draw bounding boxes or text (e.g., "Plastic: 95%") on the video feed.

## 👤 Member 5: Project Lead & Analyst

- **Goal:** Evaluate performance and write the report.
- **Primary Files:** `report/report.md`, `src/evaluate.py` (New)

**Responsibilities:**

1. **Evaluation Script:** Write code to generate Confusion Matrices and calculate Accuracy/Precision/Recall.
2. **Comparison:** Compare Member 2 (KNN) vs Member 3 (SVM) results.
3. **Reporting:** Write the "Comprehensive Technical Report" in Markdown/PDF.
4. **Repo Management:** Handle Git merges and ensure the `requirements.txt` is up to date.
