import sys
import joblib
from PyQt5.QtWidgets import QApplication, QMainWindow, QComboBox, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget, QFormLayout
import numpy as np

class DiseasePredictionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Disease Prediction")

        # Load models with error handling
        try:
            self.models = {
                "Heart Disease": joblib.load("/Users/shikarichacha/Desktop/PROJECT /models/92% model.pkl"),
                "Diabetes": joblib.load("/Users/shikarichacha/Desktop/PROJECT /models/rnn_model.pkl"),
                "Breast Cancer": joblib.load("/Users/shikarichacha/Desktop/PROJECT /models/breast cancer.pkl"),
                "Thyroid": joblib.load("/Users/shikarichacha/Desktop/PROJECT /models/thyroid_model.pkl")
            }
        except Exception as e:
            print(f"Error loading models: {e}")
            sys.exit(1)

        self.feature_labels = {
            "Heart Disease": ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"],
            "Diabetes": ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"],
            "Breast Cancer": ["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean"],
            "Thyroid": ["Age", "Gender", "Smoking", "Hx Smoking", "Hx Radiotherapy", "Thyroid Function", "Physical Examination", "Adenopathy", "Pathology", "Focality", "Risk", "T", "N", "M", "Stage", "Response"]
        }

        # Create UI elements
        self.category_label = QLabel("Select Disease Category:")
        self.category_combo = QComboBox()
        self.category_combo.addItems(self.models.keys())
        self.category_combo.currentTextChanged.connect(self.update_form)

        self.form_layout = QFormLayout()
        self.feature_inputs = {}

        self.predict_button = QPushButton("Predict")
        self.predict_button.clicked.connect(self.predict)

        # Set main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.category_label)
        main_layout.addWidget(self.category_combo)
        main_layout.addLayout(self.form_layout)
        main_layout.addWidget(self.predict_button)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Initialize form with default category
        self.update_form()

    def update_form(self):
        # Clear previous form fields
        for i in reversed(range(self.form_layout.count())):
            self.form_layout.itemAt(i).widget().setParent(None)

        # Get selected category
        category = self.category_combo.currentText()

        # Create input fields for selected category
        self.feature_inputs = {}
        for feature in self.feature_labels[category]:
            label = QLabel(feature)
            line_edit = QLineEdit()
            self.form_layout.addRow(label, line_edit)
            self.feature_inputs[feature] = line_edit

    def predict(self):
        # Get selected category and corresponding model
        category = self.category_combo.currentText()
        model = self.models[category]

        # Collect input features
        features = [float(self.feature_inputs[feature].text()) for feature in self.feature_labels[category]]
        features = np.array(features).reshape(1, -1)  # Ensure the input is 2D

        # Make prediction
        prediction = model.predict(features)

        # Display prediction with a meaningful message
        if category == "Heart Disease":
            message = "You may have a heart condition. Please consult a doctor." if prediction[0] == 1 else "You are unlikely to have a heart condition."
        elif category == "Diabetes":
            message = "You may have diabetes. Please consult a doctor." if prediction[0] == 1 else "You are unlikely to have diabetes."
        elif category == "Breast Cancer":
            message = "You may have breast cancer. Please consult a doctor." if prediction[0] == 1 else "You are unlikely to have breast cancer."
        elif category == "Thyroid":
            message = "You may have a thyroid condition. Please consult a doctor." if prediction[0] == 1 else "You are unlikely to have a thyroid condition."

        result_label = QLabel(f"Prediction: {message}")
        self.form_layout.addRow(result_label)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DiseasePredictionApp()
    window.show()
    sys.exit(app.exec_())
