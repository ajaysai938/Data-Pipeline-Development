from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer, PolynomialFeatures
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd
def numerical_transform(df):
    nums = ['float64', 'int64']
    num_attr = df.select_dtypes(include=nums).columns.tolist()
    num_pipeline = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('log', FunctionTransformer(np.log1p, validate=False)),  # log1p for numerical stability
        ('std_scaler', StandardScaler()),
        ('poly', PolynomialFeatures(2, interaction_only=False, include_bias=False))
    ])
    return num_attr, num_pipeline
def categorical_transform(df):
    cat_attr = df.select_dtypes(include=['object', 'category']).columns.tolist()
    cat_pipeline = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    return cat_attr, cat_pipeline
def full_data_pipeline(df):
    num_attr, num_pipeline = numerical_transform(df)
    cat_attr, cat_pipeline = categorical_transform(df)
    cat_attr = [col for col in cat_attr if col not in num_attr]
    full_pipeline = ColumnTransformer([
        ('num', num_pipeline, num_attr),
        ('cat', cat_pipeline, cat_attr)
    ])
    return full_pipeline
data = pd.DataFrame({
    'age': [25, np.nan, 35, 40],
    'salary': [50000, 60000, np.nan, 80000],
    'gender': ['Male', 'Female', np.nan, 'Male']
})
pipeline = full_data_pipeline(data)
transformed_data = pipeline.fit_transform(data)
transformed_df = pd.DataFrame(transformed_data)
print(transformed_df)