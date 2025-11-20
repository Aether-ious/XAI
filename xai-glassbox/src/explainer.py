import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

class ModelExplainer:
    def __init__(self, model, training_data):
        """
        Init with a trained model (Sklearn/XGBoost) and the X_train data.
        """
        self.model = model
        self.data = training_data
        
        # Initialize SHAP Explainer (TreeExplainer is fastest for XGBoost)
        # If it's a generic model, we might use KernelExplainer (slower)
        print("🧠 Initializing SHAP Explainer...")
        if isinstance(model, (xgb.XGBClassifier, xgb.XGBRegressor, xgb.Booster)):
            self.explainer = shap.TreeExplainer(model)
        else:
            # Fallback for other sklearn models
            self.explainer = shap.Explainer(model, training_data)

    def get_global_importance(self):
        """
        Returns the dataframe of feature importance.
        """
        shap_values = self.explainer(self.data)
        
        # Calculate mean absolute SHAP value for each feature
        feature_names = self.data.columns
        # Handle different SHAP output shapes
        if len(shap_values.values.shape) == 3:
            # Classification: take the positive class
            vals = np.abs(shap_values.values[:,:,1]).mean(0)
        else:
            vals = np.abs(shap_values.values).mean(0)
            
        importance_df = pd.DataFrame(list(zip(feature_names, vals)),
                                     columns=['col_name','feature_importance_vals'])
        importance_df.sort_values(by=['feature_importance_vals'],
                                  ascending=False, inplace=True)
        return importance_df

    def plot_local_explanation(self, instance_row):
        """
        Why was THIS specific row predicted this way?
        Returns a matplotlib figure.
        """
        # Calculate SHAP values for single instance
        shap_values = self.explainer(instance_row)
        
        # Waterfall plot is the best for "Why me?"
        # We need to handle the plot display context
        fig = plt.figure(figsize=(8, 6))
        
        # Check dimensions for binary classification
        if len(shap_values.values.shape) == 3:
             # Slice for the positive class (e.g., Fraud/Default)
             sv = shap_values[0, :, 1]
        else:
             sv = shap_values[0]
             
        shap.plots.waterfall(sv, show=False)
        return plt.gcf() # Get current figure

    def simulate_counterfactual(self, instance_row, feature_to_change, range_vals):
        """
        Simple What-If Analysis:
        How does the probability change if we vary ONE feature?
        """
        # Create copies of the instance
        temp_df = pd.concat([instance_row] * len(range_vals), ignore_index=True)
        temp_df[feature_to_change] = range_vals
        
        # Predict
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(temp_df)[:, 1]
        else:
            # For raw Booster
            dmatrix = xgb.DMatrix(temp_df)
            probs = self.model.predict(dmatrix)
            
        return pd.DataFrame({
            feature_to_change: range_vals,
            "New_Probability": probs
        })