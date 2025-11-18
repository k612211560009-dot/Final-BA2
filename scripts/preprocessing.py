"""
Common preprocessing utilities for all datasets
Các hàm tiền xử lý chung, tái sử dụng cho nhiều loại data
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    Lớp xử lý tiền xử lý dữ liệu chung
    Chỉ chứa các bước xử lý CHUNG, không đặc thù cho từng domain
    """
    
    def __init__(self):
        self.scaler = None
        self.imputer = None
        
    # ==================== BASIC CLEANING ====================
    
    @staticmethod
    def remove_duplicates(df, subset=None, keep='first'):
        """
        Loại bỏ dữ liệu trùng lặp
        
        Parameters:
        -----------
        df : DataFrame
        subset : list, optional - Cột để check duplicate
        keep : str - 'first', 'last', False
        
        Returns:
        --------
        DataFrame without duplicates
        """
        initial_rows = len(df)
        df_clean = df.drop_duplicates(subset=subset, keep=keep)
        removed = initial_rows - len(df_clean)
        
        if removed > 0:
            print(f"✓ Removed {removed} duplicate rows ({removed/initial_rows*100:.2f}%)")
        
        return df_clean
    
    @staticmethod
    def handle_missing_values(df, strategy='drop', threshold=0.5, fill_method='mean'):
        """
        Xử lý missing values
        
        Parameters:
        -----------
        strategy : str
            - 'drop': Drop rows/columns with missing values
            - 'fill': Fill missing values
            - 'auto': Auto decide based on threshold
        threshold : float - Ngưỡng missing để drop column (0-1)
        fill_method : str - 'mean', 'median', 'mode', 'forward', 'backward'
        
        Returns:
        --------
        DataFrame with handled missing values
        """
        df = df.copy()
        initial_rows = len(df)
        
        # Hiển thị thông tin missing
        missing_info = df.isnull().sum()
        missing_cols = missing_info[missing_info > 0]
        
        if len(missing_cols) > 0:
            print(f"\n📊 Missing values detected:")
            for col, count in missing_cols.items():
                pct = count / len(df) * 100
                print(f"   - {col}: {count} ({pct:.2f}%)")
        
        if strategy == 'drop':
            # Drop columns with too many missing values
            cols_to_drop = missing_info[missing_info / len(df) > threshold].index
            if len(cols_to_drop) > 0:
                print(f"\n✓ Dropping columns: {list(cols_to_drop)}")
                df = df.drop(columns=cols_to_drop)
            
            # Drop rows with any remaining missing values
            df = df.dropna()
            dropped = initial_rows - len(df)
            if dropped > 0:
                print(f"✓ Dropped {dropped} rows with missing values")
        
        elif strategy == 'fill':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(exclude=[np.number]).columns
            
            # Fill numeric columns
            if fill_method == 'mean':
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            elif fill_method == 'median':
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            elif fill_method == 'forward':
                df[numeric_cols] = df[numeric_cols].fillna(method='ffill')
            elif fill_method == 'backward':
                df[numeric_cols] = df[numeric_cols].fillna(method='bfill')
            
            # Fill categorical columns with mode
            for col in categorical_cols:
                if df[col].isnull().any():
                    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'unknown')
            
            print(f"✓ Filled missing values using {fill_method} method")
        
        return df
    
    @staticmethod
    def remove_outliers(df, columns=None, method='iqr', threshold=1.5):
        """
        Loại bỏ outliers
        
        Parameters:
        -----------
        columns : list - Columns to check for outliers
        method : str - 'iqr', 'zscore'
        threshold : float - IQR multiplier or Z-score threshold
        
        Returns:
        --------
        DataFrame without outliers
        """
        df = df.copy()
        initial_rows = len(df)
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        mask = pd.Series([True] * len(df))
        
        for col in columns:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR
                mask &= (df[col] >= lower) & (df[col] <= upper)
            
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                mask &= z_scores < threshold
        
        df_clean = df[mask]
        removed = initial_rows - len(df_clean)
        
        if removed > 0:
            print(f"✓ Removed {removed} outlier rows ({removed/initial_rows*100:.2f}%) using {method} method")
        
        return df_clean
    
    # ==================== SCALING ====================
    
    def fit_scaler(self, X, method='standard'):
        """
        Fit scaler to data
        
        Parameters:
        -----------
        X : array-like - Data to fit
        method : str - 'standard', 'minmax', 'robust'
        """
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        
        self.scaler.fit(X)
        print(f"✓ Fitted {method} scaler")
        return self
    
    def transform(self, X):
        """Transform data using fitted scaler"""
        if self.scaler is None:
            raise ValueError("Scaler not fitted yet. Call fit_scaler first.")
        return self.scaler.transform(X)
    
    def fit_transform(self, X, method='standard'):
        """Fit and transform in one step"""
        self.fit_scaler(X, method)
        return self.transform(X)
    
    # ==================== FEATURE ENGINEERING ====================
    
    @staticmethod
    def create_rolling_features(df, column, windows=[3, 5, 10], functions=['mean', 'std', 'min', 'max']):
        """
        Tạo rolling window features
        
        Parameters:
        -----------
        df : DataFrame
        column : str - Column to create features from
        windows : list - Window sizes
        functions : list - Aggregation functions
        
        Returns:
        --------
        DataFrame with additional rolling features
        """
        df = df.copy()
        
        for window in windows:
            for func in functions:
                col_name = f"{column}_rolling_{func}_{window}"
                if func == 'mean':
                    df[col_name] = df[column].rolling(window=window).mean()
                elif func == 'std':
                    df[col_name] = df[column].rolling(window=window).std()
                elif func == 'min':
                    df[col_name] = df[column].rolling(window=window).min()
                elif func == 'max':
                    df[col_name] = df[column].rolling(window=window).max()
        
        # Fill NaN created by rolling
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        print(f"✓ Created {len(windows) * len(functions)} rolling features for '{column}'")
        return df
    
    @staticmethod
    def create_lag_features(df, column, lags=[1, 2, 3]):
        """
        Tạo lag features
        
        Parameters:
        -----------
        df : DataFrame
        column : str - Column to create lags from
        lags : list - Lag periods
        
        Returns:
        --------
        DataFrame with lag features
        """
        df = df.copy()
        
        for lag in lags:
            df[f"{column}_lag_{lag}"] = df[column].shift(lag)
        
        # Fill NaN
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        print(f"✓ Created {len(lags)} lag features for '{column}'")
        return df
    
    # ==================== DATA SPLITTING ====================
    
    @staticmethod
    def train_test_split_time_series(df, test_size=0.2, time_column=None):
        """
        Split time series data (không shuffle)
        
        Parameters:
        -----------
        df : DataFrame
        test_size : float - Tỷ lệ test set
        time_column : str - Column to sort by (if not already sorted)
        
        Returns:
        --------
        train_df, test_df
        """
        if time_column:
            df = df.sort_values(time_column)
        
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        print(f"✓ Split data: {len(train_df)} train, {len(test_df)} test")
        return train_df, test_df
    
    # ==================== UTILITY FUNCTIONS ====================
    
    @staticmethod
    def get_data_info(df):
        """
        Hiển thị thông tin tổng quan về data
        """
        print("="*60)
        print("📊 DATA INFORMATION")
        print("="*60)
        print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"\nColumn types:")
        print(df.dtypes.value_counts())
        print(f"\nMissing values:")
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(missing[missing > 0])
        else:
            print("No missing values")
        print(f"\nDuplicate rows: {df.duplicated().sum()}")
        print("="*60)
    
    @staticmethod
    def save_processed_data(df, filepath, compression=None):
        """
        Lưu processed data
        
        Parameters:
        -----------
        filepath : str - Path to save file
        compression : str - 'gzip', 'bz2', 'zip', 'xz', None
        """
        df.to_csv(filepath, index=False, compression=compression)
        file_size = pd.read_csv(filepath).memory_usage(deep=True).sum() / 1024**2
        print(f"✓ Saved processed data to: {filepath}")
        print(f"  File size: {file_size:.2f} MB")


# ==================== QUICK FUNCTIONS ====================

def quick_clean(df, remove_duplicates=True, handle_missing='drop', 
                remove_outliers=False, outlier_method='iqr'):
    """
    Quick cleaning pipeline - Xử lý nhanh các bước cơ bản
    
    Parameters:
    -----------
    df : DataFrame
    remove_duplicates : bool
    handle_missing : str - 'drop', 'fill', None
    remove_outliers : bool
    outlier_method : str - 'iqr', 'zscore'
    
    Returns:
    --------
    Cleaned DataFrame
    """
    preprocessor = DataPreprocessor()
    
    print("\n🔧 Starting quick cleaning pipeline...")
    print("="*60)
    
    # 1. Remove duplicates
    if remove_duplicates:
        df = preprocessor.remove_duplicates(df)
    
    # 2. Handle missing values
    if handle_missing:
        df = preprocessor.handle_missing_values(df, strategy=handle_missing)
    
    # 3. Remove outliers
    if remove_outliers:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df = preprocessor.remove_outliers(df, columns=numeric_cols, method=outlier_method)
    
    print("="*60)
    print("✅ Quick cleaning completed!")
    preprocessor.get_data_info(df)
    
    return df


if __name__ == "__main__":
    print("Data Preprocessing Module")
    print("Import this module in your notebooks:")
    print("  from scripts.preprocessing import DataPreprocessor, quick_clean")
