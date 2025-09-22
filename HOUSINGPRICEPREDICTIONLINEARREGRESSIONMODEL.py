import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
class HousePricePredictor:
    def __init__(self):
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.feature_names = ['GrLivArea', 'BedroomAbvGr', 'FullBath']
        self.is_trained = False    
    def load_and_prepare_data(self, train_path='train.csv', test_path='test.csv'):
        """
        Load and prepare the Kaggle House Prices dataset
        Args:
            train_path (str): Path to training data CSV file
            test_path (str): Path to test data CSV file
        """
        try:
            # Load the training data
            self.train_data = pd.read_csv(train_path)
            print(f"Training data loaded: {self.train_data.shape}")
            # Load test data if available
            try:
                self.test_data = pd.read_csv(test_path)
                print(f"Test data loaded: {self.test_data.shape}")
            except FileNotFoundError:
                print("Test data file not found. Will use train-validation split.")
                self.test_data = None
        except FileNotFoundError:
            print("Training data file not found. Creating sample dataset...")
            self.create_sample_dataset()
        # Prepare features
        self.prepare_features()
    def create_sample_dataset(self):
        """Create a sample dataset if the original files are not available"""
        np.random.seed(42)
        n_samples = 1000
        # Generate synthetic data
        living_area = np.random.normal(1500, 500, n_samples)
        living_area = np.clip(living_area, 500, 5000)
        bedrooms = np.random.randint(1, 6, n_samples)
        bathrooms = np.random.randint(1, 4, n_samples)
        # Create realistic price based on features with some noise
        price = (living_area * 80 + bedrooms * 5000 + bathrooms * 8000 + 
                np.random.normal(0, 15000, n_samples))
        price = np.clip(price, 50000, 800000)
        self.train_data = pd.DataFrame({
            'GrLivArea': living_area,
            'BedroomAbvGr': bedrooms,
            'FullBath': bathrooms,
            'SalePrice': price
        })
        print(f"Sample dataset created: {self.train_data.shape}")
    def prepare_features(self):
        """Prepare and clean the features for modeling"""
        # Check if required columns exist
        required_cols = self.feature_names + ['SalePrice']
        available_cols = [col for col in required_cols if col in self.train_data.columns]
        if 'SalePrice' not in self.train_data.columns:
            raise ValueError("Target variable 'SalePrice' not found in training data")
        # Use available feature columns
        available_features = [col for col in self.feature_names if col in self.train_data.columns]
        self.feature_names = available_features
        print(f"Using features: {self.feature_names}")
        # Handle missing values
        for col in self.feature_names:
            if self.train_data[col].isnull().sum() > 0:
                median_val = self.train_data[col].median()
                self.train_data[col].fillna(median_val, inplace=True)
                print(f"Filled {self.train_data[col].isnull().sum()} missing values in {col}")
        # Remove outliers (optional - using IQR method)
        self.remove_outliers()
    def remove_outliers(self, factor=1.5):
        """Remove outliers using IQR method"""
        initial_shape = self.train_data.shape[0]
        for col in self.feature_names + ['SalePrice']:
            Q1 = self.train_data[col].quantile(0.25)
            Q3 = self.train_data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            self.train_data = self.train_data[
                (self.train_data[col] >= lower_bound) & 
                (self.train_data[col] <= upper_bound)
            ]
        print(f"Removed {initial_shape - self.train_data.shape[0]} outliers")
    def explore_data(self):
        """Perform exploratory data analysis"""
        print("\n" + "="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)
        # Basic statistics
        print("\nDataset Info:")
        print(f"Shape: {self.train_data.shape}")
        print(f"Features: {self.feature_names}")
        print("\nBasic Statistics:")
        print(self.train_data[self.feature_names + ['SalePrice']].describe())
        # Correlation analysis
        print("\nCorrelation with Sale Price:")
        correlations = self.train_data[self.feature_names + ['SalePrice']].corr()['SalePrice'].sort_values(ascending=False)
        print(correlations[:-1])  # Exclude self-correlation
        # Create visualizations
        self.create_visualizations()
    def create_visualizations(self):
        """Create visualizations for data exploration"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('House Price Analysis', fontsize=16, fontweight='bold')
        # Price distribution
        axes[0, 0].hist(self.train_data['SalePrice'], bins=50, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Distribution of House Prices')
        axes[0, 0].set_xlabel('Sale Price ($)')
        axes[0, 0].set_ylabel('Frequency')
        # Correlation heatmap
        corr_matrix = self.train_data[self.feature_names + ['SalePrice']].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[0, 1])
        axes[0, 1].set_title('Feature Correlation Matrix')
        # Scatter plot: Living Area vs Price
        if 'GrLivArea' in self.feature_names:
            axes[1, 0].scatter(self.train_data['GrLivArea'], self.train_data['SalePrice'], 
                             alpha=0.6, color='coral')
            axes[1, 0].set_xlabel('Above Ground Living Area (sq ft)')
            axes[1, 0].set_ylabel('Sale Price ($)')
            axes[1, 0].set_title('Living Area vs Sale Price')
        # Box plot: Bedrooms vs Price
        if 'BedroomAbvGr' in self.feature_names:
            sns.boxplot(data=self.train_data, x='BedroomAbvGr', y='SalePrice', ax=axes[1, 1])
            axes[1, 1].set_title('Bedrooms vs Sale Price')
            axes[1, 1].set_xlabel('Number of Bedrooms')
            axes[1, 1].set_ylabel('Sale Price ($)')
        plt.tight_layout()
        plt.show()
    def train_model(self, test_size=0.2, random_state=42):
        """Train the linear regression model"""
        print("\n" + "="*50)
        print("MODEL TRAINING")
        print("="*50)
        # Prepare features and target
        X = self.train_data[self.feature_names]
        y = self.train_data['SalePrice']
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        print(f"Training set size: {self.X_train.shape[0]}")
        print(f"Test set size: {self.X_test.shape[0]}")
        # Scale the features (optional but recommended)
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        # Train the model
        self.model.fit(self.X_train_scaled, self.y_train)
        self.is_trained = True
        print("Model trained successfully!")
        # Print model coefficients
        self.print_model_coefficients()
    def print_model_coefficients(self):
        """Print model coefficients and interpretation"""
        print("\nModel Coefficients:")
        print(f"Intercept: ${self.model.intercept_:,.2f}")
        for feature, coef in zip(self.feature_names, self.model.coef_):
            print(f"{feature}: ${coef:,.2f}")
        print("\nInterpretation:")
        for feature, coef in zip(self.feature_names, self.model.coef_):
            if 'Area' in feature:
                print(f"- Each additional sq ft increases price by ${coef:.2f}")
            elif 'Bedroom' in feature:
                print(f"- Each additional bedroom changes price by ${coef:,.2f}")
            elif 'Bath' in feature:
                print(f"- Each additional bathroom changes price by ${coef:,.2f}")
    def evaluate_model(self):
        """Evaluate the model performance"""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        # Make predictions
        y_train_pred = self.model.predict(self.X_train_scaled)
        y_test_pred = self.model.predict(self.X_test_scaled)
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        test_mae = mean_absolute_error(self.y_test, y_test_pred)
        train_r2 = r2_score(self.y_train, y_train_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)
        # Print metrics
        print("Performance Metrics:")
        print(f"Training RMSE: ${train_rmse:,.2f}")
        print(f"Test RMSE: ${test_rmse:,.2f}")
        print(f"Training MAE: ${train_mae:,.2f}")
        print(f"Test MAE: ${test_mae:,.2f}")
        print(f"Training R²: {train_r2:.4f}")
        print(f"Test R²: {test_r2:.4f}")
        # Create evaluation plots
        self.create_evaluation_plots(y_train_pred, y_test_pred)
        return {
            'train_rmse': train_rmse, 'test_rmse': test_rmse,
            'train_mae': train_mae, 'test_mae': test_mae,
            'train_r2': train_r2, 'test_r2': test_r2
        }
    def create_evaluation_plots(self, y_train_pred, y_test_pred):
        """Create evaluation plots"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        # Predicted vs Actual (Test Set)
        axes[0].scatter(self.y_test, y_test_pred, alpha=0.6, color='blue')
        axes[0].plot([self.y_test.min(), self.y_test.max()], 
                    [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[0].set_xlabel('Actual Price ($)')
        axes[0].set_ylabel('Predicted Price ($)')
        axes[0].set_title('Predicted vs Actual Prices (Test Set)')
        axes[0].grid(True, alpha=0.3)
        # Residuals plot
        residuals = self.y_test - y_test_pred
        axes[1].scatter(y_test_pred, residuals, alpha=0.6, color='green')
        axes[1].axhline(y=0, color='r', linestyle='--')
        axes[1].set_xlabel('Predicted Price ($)')
        axes[1].set_ylabel('Residuals ($)')
        axes[1].set_title('Residuals Plot')
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    def predict_price(self, living_area=None, bedrooms=None, bathrooms=None):
        """Make a price prediction for given house features"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        # Create feature array
        features = []
        feature_values = [living_area, bedrooms, bathrooms]
        for i, feature_name in enumerate(self.feature_names):
            if feature_values[i] is not None:
                features.append(feature_values[i])
            else:
                # Use median value if not provided
                median_val = self.train_data[feature_name].median()
                features.append(median_val)
        features_array = np.array(features).reshape(1, -1)
        features_scaled = self.scaler.transform(features_array)
        prediction = self.model.predict(features_scaled)[0]
        print(f"\nPrice Prediction:")
        print(f"Living Area: {features[0]:,.0f} sq ft")
        if len(features) > 1:
            print(f"Bedrooms: {features[1]:.0f}")
        if len(features) > 2:
            print(f"Bathrooms: {features[2]:.0f}")
        print(f"Predicted Price: ${prediction:,.2f}")
        return prediction
def main():
    """Main function to run the house price prediction pipeline"""
    # Initialize the predictor
    predictor = HousePricePredictor()
    # Load and prepare data
    print("Loading and preparing data...")
    predictor.load_and_prepare_data()
    # Explore the data
    predictor.explore_data()
    # Train the model
    predictor.train_model()
    # Evaluate the model
    metrics = predictor.evaluate_model()
    # Make sample predictions
    print("\n" + "="*50)
    print("SAMPLE PREDICTIONS")
    print("="*50)
    # Example predictions
    predictor.predict_price(living_area=2000, bedrooms=3, bathrooms=2)
    predictor.predict_price(living_area=1500, bedrooms=2, bathrooms=1)
    predictor.predict_price(living_area=3000, bedrooms=4, bathrooms=3)
    return predictor
if __name__ == "__main__":
    # Run the main pipeline
    house_predictor = main()
    print("\n" + "="*50)
    print("PIPELINE COMPLETED!")
    print("="*50)
    print("You can now use the 'house_predictor' object to make predictions:")
    print("house_predictor.predict_price(living_area=2500, bedrooms=3, bathrooms=2)")