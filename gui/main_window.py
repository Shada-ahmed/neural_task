
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
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from algorithms.perceptron import Perceptron
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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Neural Networks Lab 3 | Perceptron")
        self.resize(1250, 950)
        self.setMinimumSize(950, 700)

        self.data = preprocess_data(load_data(DATA_PATH))
        self.experiment_history = []

        self.create_widgets()
        self.populate_dropdowns()
        self.apply_style()

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
        title = QLabel("Neural Networks Lab 3")
        title.setObjectName("pageTitle")

        subtitle = QLabel(
            "Penguin Classification  |  Perceptron Training and Evaluation"
        )
        subtitle.setObjectName("subtitle")

        main_layout.addWidget(title)
        main_layout.addWidget(subtitle)

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

        settings = [
            ("Feature 1", self.feature1_combo, 0, 0),
            ("Feature 2", self.feature2_combo, 0, 2),
            ("Class 1 (-1)", self.class1_combo, 1, 0),
            ("Class 2 (+1)", self.class2_combo, 1, 2),
            ("Learning rate", self.learning_rate_combo, 2, 0),
            ("Maximum epochs", self.epochs_combo, 2, 2),
            ("Use bias", self.bias_combo, 3, 0),
        ]

        for label_text, combo, row, column in settings:
            label = QLabel(label_text)
            label.setMinimumWidth(95)

            settings_layout.addWidget(label, row, column)
            settings_layout.addWidget(combo, row, column + 1)

        settings_layout.setColumnStretch(1, 1)
        settings_layout.setColumnStretch(3, 1)

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

        self.confusion_table.setHorizontalHeaderLabels(
            ["Predicted -1", "Predicted +1"]
        )
        self.confusion_table.setVerticalHeaderLabels(
            ["Actual -1", "Actual +1"]
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

        self.history_table = QTableWidget(0, 8)

        self.history_table.setHorizontalHeaderLabels([
            "Experiment",
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

            X_train, X_test, y_train, y_test = prepare_data(
                data=self.data,
                feature1=feature1,
                feature2=feature2,
                class1=class1,
                class2=class2,
                train_samples=30,
                test_samples=20,
                random_state=42,
            )

            model = Perceptron(
                learning_rate=learning_rate,
                epochs=epochs,
                use_bias=use_bias,
            )

            model.fit(X_train, y_train)

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

            errors_per_epoch = list(
                getattr(model, "errors_per_epoch", [])
            )

            self.show_results(
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
                errors_per_epoch=errors_per_epoch,
            )

        except Exception as error:
            QMessageBox.critical(
                self,
                "Training Error",
                str(error),
            )

    # =========================================================
    # RESULTS
    # =========================================================

    def show_results(
        self,
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
        errors_per_epoch,
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
            f"Epochs Used\n{len(errors_per_epoch)}"
        )
        self.assessment_label.setText(
            f"Assessment: {assessment}"
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
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            feature1=feature1,
            feature2=feature2,
            class1=class1,
            class2=class2,
            errors_per_epoch=errors_per_epoch,
        )

        experiment = {
            "experiment": len(self.experiment_history) + 1,
            "feature_1": feature1,
            "feature_2": feature2,
            "class_1": class1,
            "class_2": class2,
            "learning_rate": learning_rate,
            "requested_epochs": requested_epochs,
            "use_bias": use_bias,
            "training_accuracy_percent": round(
                float(train_accuracy), 2
            ),
            "testing_accuracy_percent": round(
                float(test_accuracy), 2
            ),
            "accuracy_gap_percentage_points": round(
                float(accuracy_gap), 2
            ),
            "epochs_used": len(errors_per_epoch),
            "actual_-1_predicted_-1": int(matrix[0, 0]),
            "actual_-1_predicted_+1": int(matrix[0, 1]),
            "actual_+1_predicted_-1": int(matrix[1, 0]),
            "actual_+1_predicted_+1": int(matrix[1, 1]),
            "bias_value": float(model.bias),
            "weight_1": float(model.weights[0]),
            "weight_2": float(model.weights[1]),
        }

        self.experiment_history.append(experiment)
        self.add_history_row(experiment)

    # =========================================================
    # PLOTS
    # =========================================================

    
    def plot_results(
        self,
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        feature1,
        feature2,
        class1,
        class2,
        errors_per_epoch,
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

        # Training errors per epoch.
        if errors_per_epoch:
            error_ax.plot(
                range(1, len(errors_per_epoch) + 1),
                errors_per_epoch,
                color="#7C3AED",
                marker="o",
                markersize=5,
                linewidth=2,
            )

            error_ax.set_xlabel("Epoch")
            error_ax.set_ylabel("Misclassified training samples")
            error_ax.set_title("Training Errors per Epoch")
            error_ax.set_ylim(
                bottom=0,
                top=max(errors_per_epoch) + 1,
            )
            error_ax.grid(True, alpha=0.25)

        else:
            error_ax.text(
                0.5,
                0.5,
                "No training-error history available",
                transform=error_ax.transAxes,
                ha="center",
                va="center",
            )
            error_ax.set_title("Training Errors per Epoch")

        self.canvas.draw()


    # =========================================================
    # EXPERIMENT HISTORY
    # =========================================================

    def add_history_row(self, experiment):
        row = self.history_table.rowCount()
        self.history_table.insertRow(row)

        values = [
            experiment["experiment"],
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
                "Train the Perceptron at least once before exporting.",
            )
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Experiment Results",
            str(PROJECT_ROOT / "perceptron_experiments.csv"),
            "CSV Files (*.csv)",
        )

        if not file_path:
            return

        if not file_path.lower().endswith(".csv"):
            file_path += ".csv"

        try:
            fieldnames = list(self.experiment_history[0].keys())

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
