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