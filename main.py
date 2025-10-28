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
    
    def analyze_by_category(self):
        """Analyze spending by category."""
        print("\n" + "=" * 50)
        print("Category Analysis")
        print("=" * 50)
        
        # Separate income and expense
        expenses = self.df[self.df['amount'] < 0].copy()
        income = self.df[self.df['amount'] > 0].copy()
        
        # Category summary by expenses
        self.category_summary = expenses.groupby('category').agg(
            {
                'abs_amount': ['sum', 'mean', 'count'],
                'description': 'count'
            }
        ).round(2)
        
        self.category_summary.columns = ['total_spent', 'avg_transaction',
                                         'transaction_count', 'descriptions']
        
        self.category_summary = self.category_summary.sort_values(
            'total_spent', ascending=False
        )
        
        print("\nExpense summary by category: ")
        print(self.category_summary)
        
        # Calculate percentage of total spending
        total_expense = expenses['abs_amount'].sum()
        self.category_summary['percentage'] = (
            (self.category_summary['total_spent'] / total_expense) * 100
        ).round(2)
        
        print("\nTop 5 Spending Categories: ")
        print(self.category_summary.head(5))
        
    def calculate_monthly_balance(self):
        """Calculate monthly income, expenses, and balance."""
        print("\n" + "=" * 50)
        print("Monthly Balance Analysis")
        print("=" * 50)
        
        # Group by year and month
        self.monthly_summary = self.df.groupby(['year_month']).agg(
            {
                'amount': [
                    lambda x: x[x > 0].sum(),  # Total income
                    lambda x: x[x < 0].sum(),  # Total expenses
                    'sum'                      # Net balance
                ]
            }
        ).round(2)
        
        self.monthly_summary.columns = ['total_income', 'total_expenses', 'net_balance']
        
        # Calculate absolute values for expenses
        self.monthly_summary['total_expenses_abs'] = self.monthly_summary['total_expenses'].abs()
        
        # Calculate savings rate
        self.monthly_summary['savings_rate'] = (
            (self.monthly_summary['net_balance'] / self.monthly_summary['total_income']) * 100
        ).round(2)
        
        # Calculate cumulative balance
        self.monthly_summary['cumulative_balance'] = self.monthly_summary['net_balance'].cumsum()
        
        print("\nMonthly Summary: ")
        print(self.monthly_summary)
        
        # Overall statistics
        print("\n" + "=" * 50)
        print("Overall Statistics")
        print("=" * 50)
        print(f"Average Monthly Income: ${self.monthly_summary['total_income'].mean():,.2f}")
        print(f"Average Monthly Expenses: ${self.monthly_summary['total_expenses_abs'].mean():,.2f}")
        print(f"Average Monthly Savings: ${self.monthly_summary['net_balance'].mean():,.2f}")
        print(f"Average Savings Rate: ${self.monthly_summary['savings_rate'].mean():,.2f}%")
        
        return self
    
    def analyze_spending_patterns(self):
        """Analyze spending patterns by day of week and time."""
        print("\n" + "=" * 50)
        print("Spending Patterns Analysis")
        print("=" * 50)
        
        expenses = self.df[self.df['amount'] < 0].copy()
        
        # Spending by day of week
        day_spending = expenses.groupby('day_of_week')['abs_amount'].agg(
            [
                'sum', 'mean', 'count'
            ]
        ).round(2)
        day_spending.columns = ['total', 'average', 'count']
        
        # Reorder days
        days_order = ['monday', 'tuesday', 'wednesday', 'thursday',
                      'friday', 'saturday', 'sunday']
        day_spending = day_spending.reindex(days_order)
        
        print("\nSpending by Day of Week: ")
        print(day_spending)
        
        # Most expensive transaction per category
        print("\nMost Expensive Transaction per Category: ")
        most_expensive = expenses.loc[
            expenses.groupby('category')['abs_amount'].idxmax()
        ][['category', 'description', 'abs_amount', 'date']]
        print(most_expensive)
        
        return self
    
    def filter_transactions(self, **kwargs):
        """
        Filter transactions based on various criteria.
        
        Args:
        **kwargs: Filter parameters (category, min_amount, max_amount, 
        start_date, end_date, transaction_type, description_contains)
        
        Returns:
        DataFrame Filtered transactions
        """
        filtered = self.df.copy()
        
        # Filter by category
        if 'category' in kwargs:
            filtered = filtered[filtered['category'] == kwargs['category']]
            
        # Filter by amount range
        if 'min_amount' in kwargs:
            filtered = filtered[filtered['amount'] >= kwargs['min_amount']]
        if 'max_amount' in kwargs:
            filtered = filtered[filtered['amount'] <= kwargs['max_amount']]
            
        # Filter by date range
        if 'start_date' in kwargs:
            start = pd.to_datetime(kwargs['start_date'])
            filtered = filtered[filtered['date'] >= start]
        if 'end_date' in kwargs:
            end = pd.to_datetime(kwargs['end_date'])
            filtered = filtered[filtered['date'] <= end]
            
        # Filter by transaction type
        if 'transaction_type' in kwargs:
            filtered = filtered[filtered['transaction_type'] == kwargs['transaction_type']]
            
        # Filter by description
        if 'description_contains' in kwargs:
            filtered = filtered[
                filtered['description'].str.contains(
                    kwargs['description_contains'],
                    case=False,
                    na=False
                )
            ]
        
        return filtered