import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# 1. Load dataset
data = pd.read_csv("dataset/House_Price_Data.csv")


# 2. Features and target
X = data[["bhk", "propertytype", "location", "sqft"]]
y = data["totalprice"]


# 3. Categorical columns
categorical_features = ["propertytype", "location"]
numeric_features = ["bhk", "sqft"]


# 4. Convert text columns into numbers
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("numeric", "passthrough", numeric_features)
    ]
)


# 5. Create ML model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)


# 6. Create pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# 7. Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 8. Train model
pipeline.fit(X_train, y_train)


# 9. Test model
predictions = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)


print("Model Training Completed!")
print("Mean Absolute Error:", mae)
print("R2 Score:", r2)


# 10. Save trained model
joblib.dump(pipeline, "model/house_price_model.pkl")

print("Model saved successfully!")