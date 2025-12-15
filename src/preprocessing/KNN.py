import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Normalizer


features = "features"
raw_features = "raw_features"

X, y = [], []
rawX, rawy = [], []

label_map = {}
label_counter = 0

def load_data(base_dir, X, y):
    global label_counter
    for category in sorted(os.listdir(base_dir)):
        category_path = os.path.join(base_dir, category)
        if not os.path.isdir(category_path):
            continue

        if category not in label_map:
            label_map[category] = label_counter
            label_counter += 1

        label = label_map[category]

        for file in os.listdir(category_path):
            if file.endswith(".npy"):
                feature = np.load(os.path.join(category_path, file))
                X.append(feature)
                y.append(label)


load_data("features", X, y)
load_data("raw_features", rawX, rawy)
x_raw = np.array(rawX)
y_raw = np.array(rawy)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


scaler = Normalizer()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_test)

knn = KNeighborsClassifier(
    n_neighbors=1,
    metric="cosine",
    weights="distance"
)
knn.fit(X_train, y_train)
prediction = knn.predict(X_val)
accuracy = accuracy_score(y_test, prediction)

print(f"accuracy k = 1 : {accuracy:.4f}")