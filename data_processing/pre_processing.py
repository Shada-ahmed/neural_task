
import numpy as np
import pandas as pd


def load_data(file_path):
    """Load the Penguins dataset without dropping rows."""
    return pd.read_csv(file_path)


def preprocess_data(data):
    """
    Handle missing values and convert OriginLocation
    into numerical values.
    """
    data = data.copy()

    # Fill missing numeric values with the column median.
    numeric_columns = data.select_dtypes(include=np.number).columns

    for column in numeric_columns:
        data[column] = data[column].fillna(data[column].median())

    # Fill missing categorical values with the most frequent value.
    categorical_columns = data.select_dtypes(
        include=["object", "category"]
    ).columns

    for column in categorical_columns:
        if data[column].isna().any():
            mode = data[column].mode()

            if not mode.empty:
                data[column] = data[column].fillna(mode.iloc[0])
            else:
                data[column] = data[column].fillna("Unknown")

    # Convert OriginLocation into numerical values.
    if "OriginLocation" in data.columns:
        data["OriginLocation"] = pd.factorize(
            data["OriginLocation"]
        )[0]

    return data


def prepare_data( data, feature1, feature2, class1,class2,train_samples=30,test_samples=20, random_state=42):
    """
    Select two features and two classes, then split each
    class into training and testing samples.

    Returns:
        X_train, X_test, y_train, y_test
    """

    if feature1 == feature2:
        raise ValueError("Please select two different features.")

    if class1 == class2:
        raise ValueError("Please select two different classes.")

    if "Species" not in data.columns:
        raise ValueError("The dataset must contain a Species column.")

    if feature1 not in data.columns or feature2 not in data.columns:
        raise ValueError("One or both selected features do not exist.")

    # Keep only the two selected classes.
    selected_data = data[
        data["Species"].isin([class1, class2])
    ].copy()

    rng = np.random.default_rng(random_state)

    train_indices = []
    test_indices = []

    # Split each class separately to preserve the required counts.
    for species in [class1, class2]:
        species_indices = selected_data.index[
            selected_data["Species"] == species
        ].to_numpy()

        if len(species_indices) < train_samples + test_samples:
            raise ValueError(
                f"{species} needs at least "
                f"{train_samples + test_samples} samples."
            )

        species_indices = rng.permutation(species_indices)

        train_indices.extend(species_indices[:train_samples])
        test_indices.extend(
            species_indices[
                train_samples:train_samples + test_samples
            ]
        )

    # Select features and their corresponding class labels.
    X_train = selected_data.loc[
        train_indices, [feature1, feature2]
    ].to_numpy(dtype=float)

    X_test = selected_data.loc[
        test_indices, [feature1, feature2]
    ].to_numpy(dtype=float)

    y_train = selected_data.loc[train_indices, "Species"].map(
        {class1: -1, class2: 1}
    ).to_numpy(dtype=int)

    y_test = selected_data.loc[test_indices, "Species"].map(
        {class1: -1, class2: 1}
    ).to_numpy(dtype=int)

    # Scale using training data only.
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_test, y_train, y_test
