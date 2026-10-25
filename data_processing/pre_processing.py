import numpy as np
import pandas as pd


def load_data(file_path):
    return pd.read_csv(file_path)


def preprocess_data(data):
    return data.copy()


def prepare_data(data,feature1,feature2,class1,class2,train_samples=30,test_samples=20,random_state=42,return_scaler=False):
    """
    Select two features and two classes, then split each
    class into training and testing samples.

    Returns:
        X_train, X_test, y_train, y_test
    """

    if feature1==feature2:
        raise ValueError("Please select two different features.")

    if class1==class2:
        raise ValueError("Please select two different classes.")

    if "Species" not in data.columns:
        raise ValueError("The dataset must contain a Species column.")

    if feature1 not in data.columns or feature2 not in data.columns:
        raise ValueError("One or both selected features do not exist.")

    # Keep only the two selected classes.
    selected_data=data[
        data["Species"].isin([class1,class2])
    ].copy()

    rng=np.random.default_rng(random_state)

    train_indices=[]
    test_indices=[]

    # Split each class separately to preserve the required counts.
    for species in[class1,class2]:
        species_indices=selected_data.index[
            selected_data["Species"]==species
        ].to_numpy()

        if len(species_indices)<train_samples+test_samples:
            raise ValueError(
                f"{species}needs at least "
                f"{train_samples+test_samples}samples."
            )

        species_indices=rng.permutation(species_indices)

        train_indices.extend(species_indices[:train_samples])
        test_indices.extend(
            species_indices[
                train_samples:train_samples+test_samples
            ]
        )

    # Select features and their corresponding class labels.
    X_train=selected_data.loc[
        train_indices,[feature1,feature2]
    ].copy()

    X_test=selected_data.loc[
        test_indices,[feature1,feature2]
    ].copy()

    # Fill missing values using the training samples only.
    for column in[feature1,feature2]:
        if column=="OriginLocation":
            mode=X_train[column].mode()

            if not mode.empty:
                fill_value=mode.iloc[0]
            else:
                fill_value="Unknown"

            X_train[column]=X_train[column].fillna(fill_value)
            X_test[column]=X_test[column].fillna(fill_value)

            # Convert OriginLocation to numbers using training categories only.
            if column=="OriginLocation":
                categories=pd.unique(X_train[column])
                location_map={
                    location:index for index,location in enumerate(categories)
                }
                X_train[column]=X_train[column].map(location_map)
                X_test[column]=X_test[column].map(location_map).fillna(-1)
            elif not pd.api.types.is_numeric_dtype(X_train[column]):
                categories=pd.unique(X_train[column])
                category_map={
                    value:index for index,value in enumerate(categories)
                }
                X_train[column]=X_train[column].map(category_map)
                X_test[column]=X_test[column].map(category_map).fillna(-1)
        else:
            median=X_train[column].median()
            X_train[column]=X_train[column].fillna(median)
            X_test[column]=X_test[column].fillna(median)

    X_train=X_train.to_numpy(dtype=float)
    X_test=X_test.to_numpy(dtype=float)

    y_train=selected_data.loc[train_indices,"Species"].map(
        {class1:-1,class2:1}
    ).to_numpy(dtype=int)

    y_test=selected_data.loc[test_indices,"Species"].map(
        {class1:-1,class2:1}
    ).to_numpy(dtype=int)

    # Scale using training data only.
    mean=X_train.mean(axis=0)
    std=X_train.std(axis=0)
    std[std==0]=1
    X_train=(X_train-mean)/std
    X_test=(X_test-mean)/std

    if return_scaler:
        return X_train,X_test,y_train,y_test,mean,std

    return X_train,X_test,y_train,y_test
