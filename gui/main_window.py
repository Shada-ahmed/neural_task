import csv
from pathlib import Path

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QComboBox,
    QGroupBox,
    QGridLayout,
    QVBoxLayout,
    QHBoxLayout,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QFileDialog,
    QHeaderView,
    QAbstractItemView,
    QScrollArea,
    QRadioButton,
    QButtonGroup,
    QDoubleSpinBox,
    QLineEdit,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from algorithms.perceptron import Perceptron
from algorithms.adaline import Adaline
from data_processing.pre_processing import (
    load_data,
    preprocess_data,
    prepare_data,
)
from evaluation.metrics import (
    confusion_matrix_manual,
    accuracy_score_manual,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "penguins.csv"

ALGORITHM_PERCEPTRON = "Perceptron (SLP)"
ALGORITHM_ADALINE = "ADALINE"

# Same columns for every experiment so both algorithms can share one
# history list and one CSV file. ADALINE-only fields are "" for Perceptron.
EXPERIMENT_FIELDS = [
    "experiment",
    "algorithm",
    "feature_1",
    "feature_2",
    "class_1",
    "class_2",
    "learning_rate",
    "requested_epochs",
    "epochs_used",
    "use_bias",
    "mse_threshold",
    "training_accuracy_percent",
    "testing_accuracy_percent",
    "accuracy_gap_percentage_points",
    "actual_-1_predicted_-1",
    "actual_-1_predicted_+1",
    "actual_+1_predicted_-1",
    "actual_+1_predicted_+1",
    "weight_1",
    "weight_2",
    "bias_value",
    "final_mse",
]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Neural Networks Task 1 | Perceptron and ADALINE")
        self.resize(1250, 950)
        self.setMinimumSize(950, 700)

        self.data = preprocess_data(load_data(DATA_PATH))
        self.feature_ranges = {}
        for column in self.data.select_dtypes(include=np.number).columns:
            if column != "Species":
                self.feature_ranges[column] = (
                    float(self.data[column].min()),
                    float(self.data[column].max()),
                )
        self.experiment_history = []
        self.trained_model = None
        self.trained_class1 = None
        self.trained_class2 = None
        self.training_mean = None
        self.training_std = None
        self.trained_feature1 = None
        self.trained_feature2 = None

        self.create_widgets()
        self.populate_dropdowns()
        self.apply_style()
        self.on_algorithm_changed()

    # =========================================================
    # GUI SETUP
    # =========================================================

    def create_widgets(self):
        # Scroll area allows access to every section on smaller screens.
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)
        self.setCentralWidget(self.scroll_area)

        central_widget = QWidget()
        self.scroll_area.setWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(24, 20, 24, 24)
        main_layout.setSpacing(16)

        # Header
        title = QLabel("Neural Networks Task 1")
        title.setObjectName("pageTitle")

        subtitle = QLabel(
            "Penguin Classification  |  Perceptron and ADALINE Training and Evaluation"
        )
        subtitle.setObjectName("subtitle")

        main_layout.addWidget(title)
        main_layout.addWidget(subtitle)

        # -----------------------------------------------------
        # Algorithm selection
        # -----------------------------------------------------

        algorithm_box = QGroupBox("00  |  Algorithm")
        algorithm_layout = QHBoxLayout(algorithm_box)
        algorithm_layout.setContentsMargins(18, 24, 18, 14)
        algorithm_layout.setSpacing(24)

        self.perceptron_radio = QRadioButton("Single-Layer Perceptron (SLP)")
        self.adaline_radio = QRadioButton("ADALINE")
        self.perceptron_radio.setChecked(True)

        # QButtonGroup makes the two radio buttons mutually exclusive.
        self.algorithm_group = QButtonGroup(self)
        self.algorithm_group.addButton(self.perceptron_radio)
        self.algorithm_group.addButton(self.adaline_radio)
        self.perceptron_radio.toggled.connect(self.on_algorithm_changed)

        algorithm_layout.addWidget(self.perceptron_radio)
        algorithm_layout.addWidget(self.adaline_radio)
        algorithm_layout.addStretch(1)

        main_layout.addWidget(algorithm_box)

        # -----------------------------------------------------
        # Model configuration
        # -----------------------------------------------------

        settings_box = QGroupBox("01  |  Model Configuration")
        settings_layout = QGridLayout(settings_box)
        settings_layout.setContentsMargins(18, 24, 18, 18)
        settings_layout.setHorizontalSpacing(16)
        settings_layout.setVerticalSpacing(14)

        self.feature1_combo = QComboBox()
        self.feature2_combo = QComboBox()
        self.class1_combo = QComboBox()
        self.class2_combo = QComboBox()
        self.learning_rate_combo = QComboBox()
        self.epochs_combo = QComboBox()
        self.bias_combo = QComboBox()

        # ADALINE only. Minimum 0 blocks negative values.
        self.mse_threshold_input = QDoubleSpinBox()
        self.mse_threshold_input.setDecimals(6)
        self.mse_threshold_input.setRange(0.0, 1000.0)
        self.mse_threshold_input.setSingleStep(0.005)
        self.mse_threshold_input.setValue(0.01)
        self.mse_threshold_input.setKeyboardTracking(False)

        settings = [
            ("Feature 1", self.feature1_combo, 0, 0),
            ("Feature 2", self.feature2_combo, 0, 2),
            ("Class 1 (-1)", self.class1_combo, 1, 0),
            ("Class 2 (+1)", self.class2_combo, 1, 2),
            ("Learning rate", self.learning_rate_combo, 2, 0),
            ("Maximum epochs", self.epochs_combo, 2, 2),
            ("Use bias", self.bias_combo, 3, 0),
            ("MSE threshold", self.mse_threshold_input, 3, 2),
        ]

        self.mse_threshold_label = None

        for label_text, combo, row, column in settings:
            label = QLabel(label_text)
            label.setMinimumWidth(95)

            if combo is self.mse_threshold_input:
                self.mse_threshold_label = label

            settings_layout.addWidget(label, row, column)
            settings_layout.addWidget(combo, row, column + 1)

        settings_layout.setColumnStretch(1, 1)
        settings_layout.setColumnStretch(3, 1)

        # Random seed option
        seed_label = QLabel("Use random seed?")
        self.seed_yes_radio = QRadioButton("Yes")
        self.seed_no_radio = QRadioButton("No")
        self.seed_yes_radio.setChecked(True)

        self.seed_group = QButtonGroup(self)
        self.seed_group.addButton(self.seed_yes_radio)
        self.seed_group.addButton(self.seed_no_radio)

        seed_layout = QHBoxLayout()
        seed_layout.addWidget(seed_label)
        seed_layout.addWidget(self.seed_yes_radio)
        seed_layout.addWidget(self.seed_no_radio)
        seed_layout.addStretch(1)
        settings_layout.addLayout(seed_layout, 4, 0, 1, 4)

        main_layout.addWidget(settings_box)

        # -----------------------------------------------------
        # Action buttons
        # -----------------------------------------------------

        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(12)

        self.train_button = QPushButton("Train and Evaluate")
        self.train_button.setObjectName("primaryButton")
        self.train_button.clicked.connect(self.train_model)

        self.reset_button = QPushButton("Reset Results")
        self.reset_button.clicked.connect(self.reset_results)

        self.export_button = QPushButton("Export Experiments to CSV")
        self.export_button.clicked.connect(self.export_csv)

        buttons_layout.addWidget(self.train_button, 1)
        buttons_layout.addWidget(self.reset_button, 1)
        buttons_layout.addWidget(self.export_button, 1)

        main_layout.addLayout(buttons_layout)

        # -----------------------------------------------------
        # Classify a single sample
        # -----------------------------------------------------

        sample_box = QGroupBox("Classify a Single Sample")
        sample_layout = QGridLayout(sample_box)
        sample_layout.setContentsMargins(18, 24, 18, 18)
        sample_layout.setHorizontalSpacing(16)
        sample_layout.setVerticalSpacing(10)

        self.sample_feature1_label = QLabel("Feature 1:")
        self.sample_feature1_input = QLineEdit()
        self.sample_feature1_input.setPlaceholderText("Enter Feature 1 value")

        self.sample_feature2_label = QLabel("Feature 2:")
        self.sample_feature2_input = QLineEdit()
        self.sample_feature2_input.setPlaceholderText("Enter Feature 2 value")

        self.classify_sample_button = QPushButton("Classify Sample")
        self.classify_sample_button.clicked.connect(self.classify_single_sample)

        self.sample_prediction_label = QLabel("Prediction: Train a model first.")
        self.sample_prediction_label.setWordWrap(True)

        sample_layout.addWidget(self.sample_feature1_label, 0, 0)
        sample_layout.addWidget(self.sample_feature1_input, 0, 1)
        sample_layout.addWidget(self.sample_feature2_label, 1, 0)
        sample_layout.addWidget(self.sample_feature2_input, 1, 1)
        sample_layout.addWidget(self.classify_sample_button, 2, 0, 1, 2)
        sample_layout.addWidget(self.sample_prediction_label, 3, 0, 1, 2)

        main_layout.addWidget(sample_box)

        # -----------------------------------------------------
        # Performance metrics
        # -----------------------------------------------------

        results_box = QGroupBox("02  |  Model Performance")
        results_layout = QGridLayout(results_box)
        results_layout.setContentsMargins(16, 24, 16, 16)
        results_layout.setHorizontalSpacing(12)
        results_layout.setVerticalSpacing(12)

        self.train_accuracy_label = self.create_metric_label(
            "Training Accuracy", "—"
        )
        self.test_accuracy_label = self.create_metric_label(
            "Testing Accuracy", "—"
        )
        self.gap_label = self.create_metric_label(
            "Accuracy Gap", "—"
        )
        self.epochs_used_label = self.create_metric_label(
            "Epochs Used", "—"
        )

        results_layout.addWidget(self.train_accuracy_label, 0, 0)
        results_layout.addWidget(self.test_accuracy_label, 0, 1)
        results_layout.addWidget(self.gap_label, 0, 2)
        results_layout.addWidget(self.epochs_used_label, 0, 3)

        self.assessment_label = QLabel(
            "Assessment: Run an experiment"
        )
        self.assessment_label.setObjectName("assessment")
        self.assessment_label.setWordWrap(True)

        results_layout.addWidget(
            self.assessment_label, 1, 0, 1, 4
        )

        for column in range(4):
            results_layout.setColumnStretch(column, 1)

        main_layout.addWidget(results_box)

        # -----------------------------------------------------
        # Confusion matrix
        # -----------------------------------------------------

        matrix_box = QGroupBox("03  |  Test Confusion Matrix")
        matrix_layout = QVBoxLayout(matrix_box)
        matrix_layout.setContentsMargins(14, 24, 14, 14)

        self.confusion_table = QTableWidget(2, 2)

        # The class names are updated after each experiment.
        self.confusion_table.setHorizontalHeaderLabels(
            ["Predicted: Class 1", "Predicted: Class 2"]
        )
        self.confusion_table.setVerticalHeaderLabels(
            ["Actual: Class 1", "Actual: Class 2"]
        )

        self.confusion_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.confusion_table.verticalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )

        self.confusion_table.verticalHeader().setMinimumWidth(100)
        self.confusion_table.verticalHeader().setDefaultSectionSize(38)

        self.confusion_table.setMinimumHeight(140)
        self.confusion_table.setMaximumHeight(170)

        self.confusion_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )

        self.confusion_table.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.confusion_table.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        matrix_layout.addWidget(self.confusion_table)
        main_layout.addWidget(matrix_box)

        # -----------------------------------------------------
        # Visualization
        # -----------------------------------------------------

        charts_box = QGroupBox("04  |  Visualization")
        charts_layout = QVBoxLayout(charts_box)
        charts_layout.setContentsMargins(12, 24, 12, 12)

        self.figure = Figure(
            figsize=(12, 5.5),
            dpi=100,
            facecolor="#FFFFFF",
            constrained_layout=True,
        )

        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(390)

        charts_layout.addWidget(self.canvas)
        main_layout.addWidget(charts_box)

        # -----------------------------------------------------
        # Experiment history
        # -----------------------------------------------------

        history_box = QGroupBox("05  |  Experiment History")
        history_layout = QVBoxLayout(history_box)
        history_layout.setContentsMargins(12, 24, 12, 12)

        self.history_table = QTableWidget(0, 9)

        self.history_table.setHorizontalHeaderLabels([
            "Experiment",
            "Algorithm",
            "Features",
            "Classes",
            "Learning Rate",
            "Epochs",
            "Train Accuracy",
            "Test Accuracy",
            "Gap",
        ])

        self.history_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.history_table.horizontalHeader().setStretchLastSection(True)

        self.history_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.history_table.setAlternatingRowColors(True)
        self.history_table.setMinimumHeight(150)
        self.history_table.setMaximumHeight(220)

        history_layout.addWidget(self.history_table)
        main_layout.addWidget(history_box)

        main_layout.addStretch(1)

    def create_metric_label(self, heading, value):
        label = QLabel(f"{heading}\n{value}")
        label.setObjectName("metricCard")
        label.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        label.setWordWrap(True)
        label.setMinimumHeight(88)
        return label

    # =========================================================
    # STYLE
    # =========================================================

    def apply_style(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #F4F7FB;
            }

            QWidget {
                color: #1E293B;
                font-family: "Segoe UI";
                font-size: 10pt;
            }

            QLabel {
                background: transparent;
                padding: 1px;
            }

            QLabel#pageTitle {
                font-size: 25pt;
                font-weight: 700;
                color: #172554;
                padding: 0px;
            }

            QLabel#subtitle {
                color: #64748B;
                font-size: 10pt;
                padding: 2px 0px 8px 0px;
            }

            QGroupBox {
                background-color: #FFFFFF;
                border: 1px solid #DCE4EF;
                border-radius: 10px;
                margin-top: 14px;
                padding-top: 12px;
                font-weight: 600;
                color: #1E3A8A;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 14px;
                padding: 0px 5px;
                background-color: #F4F7FB;
            }

            QComboBox {
                background-color: #FFFFFF;
                color: #1E293B;
                border: 1px solid #CBD5E1;
                border-radius: 6px;
                padding: 4px 8px;
                min-height: 30px;
                max-height: 34px;
            }

            QLineEdit {
                background-color: #FFFFFF;
                color: #1E293B;
                border: 1px solid #CBD5E1;
                border-radius: 6px;
                padding: 4px 8px;
                min-height: 30px;
                max-height: 34px;
            }

            QLineEdit:focus {
                border: 1px solid #2563EB;
            }

            QDoubleSpinBox {
                background-color: #FFFFFF;
                color: #1E293B;
                border: 1px solid #CBD5E1;
                border-radius: 6px;
                padding: 4px 8px;
                min-height: 30px;
                max-height: 34px;
            }

            QDoubleSpinBox:focus {
                border: 1px solid #2563EB;
            }

            QDoubleSpinBox:disabled, QLabel:disabled {
                color: #94A3B8;
                background-color: #F1F5F9;
            }

            QRadioButton {
                font-weight: 600;
                spacing: 8px;
                background: transparent;
                padding: 5px 8px;
                border-radius: 5px;
            }

            QRadioButton:checked {
                color: #1D4ED8;
                background-color: #DBEAFE;
                border: 1px solid #93C5FD;
            }

            QComboBox:hover {
                border: 1px solid #60A5FA;
            }

            QComboBox:focus {
                border: 1px solid #2563EB;
            }

            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 25px;
                border-left: 1px solid #E2E8F0;
            }

            QComboBox QAbstractItemView {
                background-color: #FFFFFF;
                color: #1E293B;
                selection-background-color: #DBEAFE;
                selection-color: #1E3A8A;
                outline: none;
                padding: 4px;
            }

            QPushButton {
                background-color: #FFFFFF;
                color: #1E293B;
                border: 1px solid #CBD5E1;
                border-radius: 7px;
                padding: 10px 12px;
                min-height: 22px;
                font-weight: 600;
            }

            QPushButton:hover {
                background-color: #EFF6FF;
                border-color: #93C5FD;
            }

            QPushButton:pressed {
                background-color: #DBEAFE;
            }

            QPushButton#primaryButton {
                background-color: #2563EB;
                color: #FFFFFF;
                border: 1px solid #2563EB;
            }

            QPushButton#primaryButton:hover {
                background-color: #1D4ED8;
            }

            QLabel#metricCard {
                background-color: #F8FAFC;
                border: 1px solid #E2E8F0;
                border-radius: 8px;
                padding: 12px;
                font-size: 10pt;
                font-weight: 600;
            }

            QLabel#assessment {
                background-color: #EFF6FF;
                border: 1px solid #DBEAFE;
                border-radius: 6px;
                padding: 10px;
                color: #1D4ED8;
                font-weight: 600;
            }

            QTableWidget {
                background-color: #FFFFFF;
                alternate-background-color: #F8FAFC;
                color: #1E293B;
                border: 1px solid #E2E8F0;
                border-radius: 5px;
                gridline-color: #E2E8F0;
            }

            QTableWidget::item {
                padding: 6px;
            }

            QHeaderView::section {
                background-color: #EFF6FF;
                color: #1E3A8A;
                padding: 8px;
                border: none;
                border-bottom: 1px solid #DBEAFE;
                font-weight: 600;
            }

            QScrollArea {
                border: none;
                background-color: #F4F7FB;
            }

            QScrollBar:vertical {
                background: #EAF0F7;
                width: 12px;
                margin: 0px;
            }

            QScrollBar::handle:vertical {
                background: #B8C7DA;
                min-height: 25px;
                border-radius: 5px;
            }

            QScrollBar::handle:vertical:hover {
                background: #94A9C4;
            }

            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

    # =========================================================
    # DROPDOWNS
    # =========================================================

    def populate_dropdowns(self):
        feature_columns = self.data.select_dtypes(
            include=np.number
        ).columns.tolist()

        feature_columns = [
            column for column in feature_columns
            if column != "Species"
        ]

        self.feature1_combo.addItems(feature_columns)
        self.feature2_combo.addItems(feature_columns)

        species = sorted(
            self.data["Species"].dropna().unique().tolist()
        )

        self.class1_combo.addItems(species)
        self.class2_combo.addItems(species)

        self.learning_rate_combo.addItems(
            ["0.001", "0.01", "0.05", "0.1", "0.5"]
        )

        self.epochs_combo.addItems(
            ["50", "100", "200", "500", "1000"]
        )

        self.bias_combo.addItems(["Yes", "No"])

        if self.feature2_combo.count() > 1:
            self.feature2_combo.setCurrentIndex(1)

        if self.class2_combo.count() > 1:
            self.class2_combo.setCurrentIndex(1)

    # =========================================================
    # ALGORITHM SELECTION
    # =========================================================

    def selected_algorithm(self):
        if self.adaline_radio.isChecked():
            return ALGORITHM_ADALINE
        return ALGORITHM_PERCEPTRON

    def on_algorithm_changed(self, *_):
        # The MSE threshold only matters for ADALINE.
        is_adaline = self.selected_algorithm() == ALGORITHM_ADALINE
        self.mse_threshold_input.setEnabled(is_adaline)
        if self.mse_threshold_label is not None:
            self.mse_threshold_label.setEnabled(is_adaline)

    # =========================================================
    # TRAINING AND EVALUATION
    # =========================================================

    def train_model(self):
        feature1 = self.feature1_combo.currentText()
        feature2 = self.feature2_combo.currentText()
        class1 = self.class1_combo.currentText()
        class2 = self.class2_combo.currentText()

        if feature1 == feature2:
            QMessageBox.warning(
                self,
                "Invalid Features",
                "Please select two different features.",
            )
            return

        if class1 == class2:
            QMessageBox.warning(
                self,
                "Invalid Classes",
                "Please select two different classes.",
            )
            return

        try:
            learning_rate = float(
                self.learning_rate_combo.currentText()
            )
            epochs = int(self.epochs_combo.currentText())
            use_bias = self.bias_combo.currentText() == "Yes"

            X_train, X_test, y_train, y_test, self.training_mean, self.training_std = prepare_data(
                data=self.data,
                feature1=feature1,
                feature2=feature2,
                class1=class1,
                class2=class2,
                train_samples=30,
                test_samples=20,
                random_state=42 if self.seed_yes_radio.isChecked() else None,
                return_scaler=True,
            )

            algorithm = self.selected_algorithm()
            mse_threshold = None

            if algorithm == ALGORITHM_ADALINE:
                mse_threshold = float(self.mse_threshold_input.value())

                if mse_threshold < 0:
                    QMessageBox.warning(
                        self,
                        "Invalid MSE Threshold",
                        "The MSE threshold must be zero or larger.",
                    )
                    return

                model = Adaline(
                    learning_rate=learning_rate,
                    epochs=epochs,
                    use_bias=use_bias,
                    random_state=42 if self.seed_yes_radio.isChecked() else None,
                    mse_threshold=mse_threshold,
                )
            else:
                model = Perceptron(
                    learning_rate=learning_rate,
                    epochs=epochs,
                    use_bias=use_bias,
                    random_state=42 if self.seed_yes_radio.isChecked() else None,
                )

            model.fit(X_train, y_train)
            print("Use bias:", use_bias)
            print("Model bias:", model.bias)
            print("Weights:", model.weights)

            self.trained_model = model
            self.trained_class1 = class1
            self.trained_class2 = class2
            self.trained_feature1 = feature1
            self.trained_feature2 = feature2

            train_predictions = model.predict(X_train)
            test_predictions = model.predict(X_test)

            train_accuracy = accuracy_score_manual(
                y_train, train_predictions
            )
            test_accuracy = accuracy_score_manual(
                y_test, test_predictions
            )

            matrix = confusion_matrix_manual(
                y_test, test_predictions
            )

            # Training history depends on the algorithm:
            # Perceptron -> misclassified samples, ADALINE -> MSE.
            if algorithm == ALGORITHM_ADALINE:
                history = list(model.msePerEpoch)
            else:
                history = list(getattr(model, "errors_per_epoch", []))

            self.show_results(
                algorithm=algorithm,
                mse_threshold=mse_threshold,
                model=model,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                feature1=feature1,
                feature2=feature2,
                class1=class1,
                class2=class2,
                learning_rate=learning_rate,
                requested_epochs=epochs,
                use_bias=use_bias,
                train_accuracy=train_accuracy,
                test_accuracy=test_accuracy,
                matrix=matrix,
                history=history,
            )

        except Exception as error:
            QMessageBox.critical(
                self,
                "Training Error",
                str(error),
            )

    def classify_single_sample(self):
        if self.trained_model is None:
            QMessageBox.information(
                self,
                "Model Not Trained",
                "Please train and evaluate a model before classifying a sample."
            )
            return

        value1_text = self.sample_feature1_input.text().strip()
        value2_text = self.sample_feature2_input.text().strip()

        if not value1_text or not value2_text:
            QMessageBox.warning(
                self,
                "Missing Input",
                "Please enter values for both features."
            )
            return

        try:
            value1 = float(value1_text)
            value2 = float(value2_text)
        except ValueError:
            QMessageBox.warning(
                self,
                "Invalid Input",
                "Please enter valid numeric values for both features."
            )
            return

        if not np.isfinite(value1) or not np.isfinite(value2):
            QMessageBox.warning(
                self,
                "Invalid Input",
                "Feature values must be finite numbers."
            )
            return

        # Check that the values fall within the observed dataset range.
        feature_ranges = getattr(self, "feature_ranges", {})
        for feature, value in [
            (self.trained_feature1, value1),
            (self.trained_feature2, value2),
        ]:
            if feature in feature_ranges:
                minimum, maximum = feature_ranges[feature]
                if value < minimum or value > maximum:
                    QMessageBox.warning(
                        self,
                        "Value Out of Range",
                        f"{feature} must be between {minimum:g} and {maximum:g}."
                    )
                    return

        sample = np.array([[value1, value2]], dtype=float)
        sample = (sample - self.training_mean) / self.training_std
        prediction = int(self.trained_model.predict(sample)[0])

        if prediction == -1:
            predicted_class = self.trained_class1
        else:
            predicted_class = self.trained_class2

        self.sample_prediction_label.setText(
            f"Prediction: {predicted_class} "
            f"({self.trained_feature1} = {value1:g}, "
            f"{self.trained_feature2} = {value2:g})"
        )

    # =========================================================
    # RESULTS
    # =========================================================

    def show_results(
        self,
        algorithm,
        mse_threshold,
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        feature1,
        feature2,
        class1,
        class2,
        learning_rate,
        requested_epochs,
        use_bias,
        train_accuracy,
        test_accuracy,
        matrix,
        history,
    ):
        accuracy_gap = train_accuracy - test_accuracy

        if accuracy_gap > 10:
            assessment = "Possible overfitting: large train/test gap"
        elif accuracy_gap > 5:
            assessment = "Moderate train/test accuracy gap"
        elif accuracy_gap < -5:
            assessment = "Test accuracy is notably higher than training"
        else:
            assessment = "Small train/test accuracy gap"

        self.train_accuracy_label.setText(
            f"Training Accuracy\n{train_accuracy:.2f}%"
        )
        self.test_accuracy_label.setText(
            f"Testing Accuracy\n{test_accuracy:.2f}%"
        )
        self.gap_label.setText(
            f"Accuracy Gap\n{accuracy_gap:+.2f}%"
        )
        self.epochs_used_label.setText(
            f"Epochs Used\n{len(history)}"
        )
        self.assessment_label.setText(
            f"Assessment: {assessment}"
        )

        # Show the selected class names instead of -1 and +1.
        self.confusion_table.setHorizontalHeaderLabels(
            [f"Predicted: {class1}", f"Predicted: {class2}"]
        )
        self.confusion_table.setVerticalHeaderLabels(
            [f"Actual: {class1}", f"Actual: {class2}"]
        )

        for row in range(2):
            for column in range(2):
                item = QTableWidgetItem(
                    str(int(matrix[row, column]))
                )
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignCenter
                )
                self.confusion_table.setItem(row, column, item)

        self.plot_results(
            algorithm=algorithm,
            mse_threshold=mse_threshold,
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            feature1=feature1,
            feature2=feature2,
            class1=class1,
            class2=class2,
            history=history,
        )

        is_adaline = algorithm == ALGORITHM_ADALINE

        experiment = {
            "experiment": len(self.experiment_history) + 1,
            "algorithm": algorithm,
            "feature_1": feature1,
            "feature_2": feature2,
            "class_1": class1,
            "class_2": class2,
            "learning_rate": learning_rate,
            "requested_epochs": requested_epochs,
            "epochs_used": len(history),
            "use_bias": use_bias,
            "mse_threshold": mse_threshold if is_adaline else "",
            "training_accuracy_percent": round(
                float(train_accuracy), 2
            ),
            "testing_accuracy_percent": round(
                float(test_accuracy), 2
            ),
            "accuracy_gap_percentage_points": round(
                float(accuracy_gap), 2
            ),
            "actual_-1_predicted_-1": int(matrix[0, 0]),
            "actual_-1_predicted_+1": int(matrix[0, 1]),
            "actual_+1_predicted_-1": int(matrix[1, 0]),
            "actual_+1_predicted_+1": int(matrix[1, 1]),
            "weight_1": float(model.weights[0]),
            "weight_2": float(model.weights[1]),
            "bias_value": float(model.bias),
            "final_mse": float(history[-1]) if is_adaline and history else "",
        }

        self.experiment_history.append(experiment)
        self.add_history_row(experiment)

    # =========================================================
    # PLOTS
    # =========================================================

    
    def plot_results(
        self,
        algorithm,
        mse_threshold,
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        feature1,
        feature2,
        class1,
        class2,
        history,
    ):
        self.figure.clear()

        boundary_ax = self.figure.add_subplot(1, 2, 1)
        error_ax = self.figure.add_subplot(1, 2, 2)

        # Display only the test samples.
        boundary_ax.scatter(
            X_test[y_test == -1, 0],
            X_test[y_test == -1, 1],
            label=f"{class1} (actual -1)",
            marker="o",
            color="#2563EB",
            s=65,
            alpha=0.9,
        )

        boundary_ax.scatter(
            X_test[y_test == 1, 0],
            X_test[y_test == 1, 1],
            label=f"{class2} (actual +1)",
            marker="x",
            color="#EA580C",
            s=65,
            linewidths=2,
            alpha=0.9,
        )

        # Draw the learned decision boundary.
        weights = np.asarray(model.weights, dtype=float)
        bias = float(model.bias)

        x_min = X_test[:, 0].min()
        x_max = X_test[:, 0].max()
        y_min = X_test[:, 1].min()
        y_max = X_test[:, 1].max()

        x_padding = max((x_max - x_min) * 0.12, 0.1)
        y_padding = max((y_max - y_min) * 0.12, 0.1)

        x_min -= x_padding
        x_max += x_padding
        y_min -= y_padding
        y_max += y_padding

        if abs(weights[1]) > 1e-12:
            x_values = np.linspace(x_min, x_max, 300)
            y_values = (
                -weights[0] * x_values - bias
            ) / weights[1]

            # Keep only the section of the line visible in the graph.
            visible = (y_values >= y_min) & (y_values <= y_max)

            boundary_ax.plot(
                x_values[visible],
                y_values[visible],
                color="#16A34A",
                linewidth=2.5,
                label="Decision boundary",
            )

        elif abs(weights[0]) > 1e-12:
            boundary_ax.axvline(
                x=-bias / weights[0],
                color="#16A34A",
                linewidth=2.5,
                label="Decision boundary",
            )
        else:
            boundary_ax.text(
                0.5,
                0.5,
                "Weights are zero",
                transform=boundary_ax.transAxes,
                ha="center",
            )

        boundary_ax.set_xlim(x_min, x_max)
        boundary_ax.set_ylim(y_min, y_max)
        boundary_ax.set_xlabel(f"{feature1} (standardized)")
        boundary_ax.set_ylabel(f"{feature2} (standardized)")
        boundary_ax.set_title("Test Samples and Decision Boundary")
        boundary_ax.grid(True, alpha=0.25)
        boundary_ax.legend(fontsize=8, loc="best")

        # Training history: errors (Perceptron) or MSE (ADALINE).
        is_adaline = algorithm == ALGORITHM_ADALINE

        if is_adaline:
            history_title = "MSE per Epoch"
            history_ylabel = "Mean squared error (MSE)"
        else:
            history_title = "Training Errors per Epoch"
            history_ylabel = "Misclassified training samples"

        if history:
            error_ax.plot(
                range(1, len(history) + 1),
                history,
                color="#7C3AED",
                marker="o",
                markersize=5,
                linewidth=2,
                label="MSE" if is_adaline else "Errors",
            )

            error_ax.set_xlabel("Epoch")
            error_ax.set_ylabel(history_ylabel)
            error_ax.set_title(history_title)
            error_ax.grid(True, alpha=0.25)

            if is_adaline:
                error_ax.axhline(
                    y=mse_threshold,
                    color="#DC2626",
                    linestyle="--",
                    linewidth=1.8,
                    label=f"MSE threshold = {mse_threshold:g}",
                )
                error_ax.set_ylim(bottom=0)
                error_ax.legend(fontsize=8, loc="best")
            else:
                error_ax.set_ylim(
                    bottom=0,
                    top=max(history) + 1,
                )

        else:
            error_ax.text(
                0.5,
                0.5,
                "No training history available",
                transform=error_ax.transAxes,
                ha="center",
                va="center",
            )
            error_ax.set_title(history_title)

        self.canvas.draw()


    # =========================================================
    # EXPERIMENT HISTORY
    # =========================================================

    def add_history_row(self, experiment):
        row = self.history_table.rowCount()
        self.history_table.insertRow(row)

        values = [
            experiment["experiment"],
            experiment["algorithm"],
            f'{experiment["feature_1"]} + {experiment["feature_2"]}',
            f'{experiment["class_1"]} vs {experiment["class_2"]}',
            experiment["learning_rate"],
            experiment["requested_epochs"],
            f'{experiment["training_accuracy_percent"]:.2f}%',
            f'{experiment["testing_accuracy_percent"]:.2f}%',
            f'{experiment["accuracy_gap_percentage_points"]:+.2f}%',
        ]

        for column, value in enumerate(values):
            item = QTableWidgetItem(str(value))
            item.setTextAlignment(
                Qt.AlignmentFlag.AlignCenter
            )
            self.history_table.setItem(row, column, item)

        self.history_table.scrollToBottom()

    # =========================================================
    # CSV EXPORT
    # =========================================================

    def export_csv(self):
        if not self.experiment_history:
            QMessageBox.information(
                self,
                "No Experiments",
                "Train a model at least once before exporting.",
            )
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Experiment Results",
            str(PROJECT_ROOT / "experiments.csv"),
            "CSV Files (*.csv)",
        )

        if not file_path:
            return

        if not file_path.lower().endswith(".csv"):
            file_path += ".csv"

        try:
            fieldnames = EXPERIMENT_FIELDS

            with open(
                file_path,
                "w",
                newline="",
                encoding="utf-8-sig",
            ) as csv_file:
                writer = csv.DictWriter(
                    csv_file,
                    fieldnames=fieldnames,
                )
                writer.writeheader()
                writer.writerows(self.experiment_history)

            QMessageBox.information(
                self,
                "Export Successful",
                f"Exported {len(self.experiment_history)} experiments to:\n"
                f"{file_path}",
            )

        except OSError as error:
            QMessageBox.critical(
                self,
                "Export Error",
                str(error),
            )

    # =========================================================
    # RESET
    # =========================================================

    def reset_results(self):
        self.train_accuracy_label.setText("Training Accuracy\n—")
        self.test_accuracy_label.setText("Testing Accuracy\n—")
        self.gap_label.setText("Accuracy Gap\n—")
        self.epochs_used_label.setText("Epochs Used\n—")

        self.assessment_label.setText(
            "Assessment: Run an experiment"
        )

        self.confusion_table.clearContents()
        self.figure.clear()
        self.canvas.draw()

        self.history_table.setRowCount(0)
        self.experiment_history.clear()