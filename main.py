# A comprehensive analysis of personal banking transaction using Pandas.

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class BankTransactionAnalyzer:
    """Main class for analyzing bank transactions."""
    
    def __init__(self, data_path):
        """
        Initialize the analyzer with transaction data.
        
        Args:
        data_path (str): path to th CSV file containing transaction data.
        """
        self.data_path = data_path
        self.df = None
        self.monthly_summary = None
        self.category_summary = None
        
    def load_data(self):
        """Load data perform initial  data cleaning."""
        print("Loading transaction data...")
        
        # Load data
        self.df = pd.read_csv(self.data_path)
        
        # Display basic info
        print(f"Loaded {len(self.df)} transactions.")
        print("\nDataset columns:", self.df.columns.tolist())
        print("\nFirst 5 rows:\n", self.df.head())
        
        return self
    
    def clean_and_prepare_data(self):
        """Clean and prepare data for analysis."""
        print("\nCleaning and preparing data...")
        
        # Convert date column to datetime
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        # Sort by date
        self.df = self.df.sort_values('date').reset_index(drop=True)
        
        # Extract date components
        self.df['year'] = self.df['date'].dt.year
        self.df['month'] = self.df['date'].dt.month
        self.df['month_name'] = self.df['date'].dt.month_name()
        self.df['day'] = self.df['date'].dt.day
        self.df['day_of_week'] = self.df['date'].dt.day_name()
        self.df['year_month'] = self.df['date'].dt.to_period('M')
        
        # Clean amount column (ensure numeric)
        self.df['amount'] = pd.to_numeric(self.df['amount'], errors='coerce')
        
        # Remove any rows with missing critical data
        self.df = self.df.dropna(subset=['date', 'amount', 'category'])
        
        # Create transaction type (income vs expense)
        self.df['transaction_type'] = self.df['amount'].apply(
            lambda x: 'income' if x > 0 else 'expense'
        )
        
        # Create absolute amount for easier analysis
        self.df['abs_amount'] = self.df['amount'].abs()
        
        print(f"Data cleaned. {len(self.df)} valid transactions remaining...")
        print(f"Data range: {self.df['date'].min().date()} to {self.df['date'].max()}")
        
        return self