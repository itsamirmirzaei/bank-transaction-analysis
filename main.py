"""
bank_analyzer.py
~~~~~~~~~~~~~~~~
A production-ready personal bank transaction analyzer.
Features:
- Auto-detects CSV in 'data/' or current directory
- Robust parsing of date, amount, and category
- Full method chaining
- Clean, maintainable, testable code
- Professional logging
- Configurable thresholds and paths
- Optional plotting (matplotlib)
"""
import os
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path
import pandas as pd
import numpy as np

# Configure professional logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


class BankTransactionAnalyzer:
    """
    A clean, professional analyzer for personal bank transactions.
    Attributes:
        data_path (Path): Path to input CSV
        output_dir (Path): Directory for output files
        df (pd.DataFrame): Processed transaction data
    """

    # Default configuration
    DEFAULT_DATA_CANDIDATES = [
        Path("data") / "transactions.csv",
        Path("transactions.csv")
    ]
    DEFAULT_OUTPUT_DIR = Path("outputs")

    def __init__(
        self,
        data_path: Optional[str | Path] = None,
        output_dir: Optional[str | Path] = None,
        *,
        z_threshold: float = 3.0,
        high_value_quantile: float = 0.90,
        dayfirst: bool = True
    ) -> None:
        """
        Initialize the analyzer.
        Args:
            data_path: Path to CSV file. Auto-detected if None.
            output_dir: Output directory. Defaults to 'outputs/'.
            z_threshold: Z-score threshold for anomaly detection.
            high_value_quantile: Quantile for high-value transactions.
            dayfirst: Parse dates with day first (e.g., 15/03/2025).
        Raises:
            FileNotFoundError: If CSV file cannot be found.
            OSError: If output directory is not writable.
        """
        self.z_threshold = z_threshold
        self.high_value_quantile = high_value_quantile
        self.dayfirst = dayfirst

        self.output_dir = Path(output_dir) if output_dir else self.DEFAULT_OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._validate_output_dir_writable()

        self.data_path = self._resolve_data_path(data_path)
        self.df: Optional[pd.DataFrame] = None

        log.info("Analyzer initialized")
        log.info(f"Data path: {self.data_path}")
        log.info(f"Output directory: {self.output_dir.resolve()}")

    # Private Helpers
    def _validate_output_dir_writable(self) -> None:
        """Ensure output directory is writable."""
        test_path = self.output_dir / ".write_test.tmp"
        try:
            test_path.touch()
            test_path.unlink()
        except OSError as e:
            raise OSError(f"Output directory is not writable: {self.output_dir}") from e

    @staticmethod
    def _resolve_data_path(user_path: Optional[str | Path]) -> Path:
        """Resolve and validate the input CSV path."""
        if user_path:
            path = Path(user_path)
            if path.exists() and path.is_file():
                return path.resolve()
            raise FileNotFoundError(f"Provided file not found or is not a file: {path}")
        for candidate in BankTransactionAnalyzer.DEFAULT_DATA_CANDIDATES:
            if candidate.exists() and candidate.is_file():
                log.info(f"Auto-detected data file: {candidate}")
                return candidate.resolve()
        raise FileNotFoundError(
            f"Transaction file not found. Searched:\n" +
            "\n".join(f"  - {p}" for p in BankTransactionAnalyzer.DEFAULT_DATA_CANDIDATES)
        )

    @staticmethod
    def _find_column(candidates: List[str], columns: pd.Index, *, required: bool = True) -> Optional[str]:
        """Find a column by partial case-insensitive match."""
        cols_lower = {col.lower(): col for col in columns}
        for cand in candidates:
            if cand.lower() in cols_lower:
                return cols_lower[cand.lower()]
        if not required:
            return None
        raise KeyError(f"Column not found. Candidates: {candidates}")

    # Core Pipeline
    def load_data(self) -> 'BankTransactionAnalyzer':
        """Load CSV data into a DataFrame."""
        log.info("Loading transaction data...")
        self.df = pd.read_csv(self.data_path)
        log.info(f"Loaded {len(self.df):,} rows")
        return self

    def clean_and_prepare_data(self) -> 'BankTransactionAnalyzer':
        """Clean, enrich, and standardize the dataset."""
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        log.info("Cleaning and preparing data...")
        df = self.df.copy()

        # Clean column names
        df.columns = [col.strip() for col in df.columns]

        # === Find and rename critical columns ===
        date_col = self._find_column(['Date'], df.columns)
        amount_col = self._find_column(['Amount', 'INR'], df.columns)
        category_col = self._find_column(['Category'], df.columns)
        type_col = self._find_column(['Income/Expense', 'Type'], df.columns, required=False)

        df = df.rename(columns={
            date_col: 'Date_raw',
            amount_col: 'amount_raw',
            category_col: 'Category_raw'
        })

        # === Parse Date ===
        df['Date'] = pd.to_datetime(df['Date_raw'], errors='coerce', dayfirst=self.dayfirst)
        df = df.dropna(subset=['Date']).copy()

        # === Parse Amount & Type ===
        df['amount_raw'] = pd.to_numeric(df['amount_raw'], errors='coerce')

        if type_col:
            df['amount'] = df.apply(
                lambda row: row['amount_raw'] if row[type_col] == 'Income'
                else -row['amount_raw'] if row[type_col] == 'Expense'
                else np.nan,
                axis=1
            )
        else:
            df['amount'] = df['amount_raw']

        # === Category ===
        df['Category'] = df['Category_raw'].astype(str).str.strip()

        # === Final cleanup ===
        before = len(df)
        df = df.dropna(subset=['Date', 'amount', 'Category']).copy()
        after = len(df)

        # === Enrich with derived fields ===
        df = df.assign(
            year=df['Date'].dt.year,
            month=df['Date'].dt.month,
            day_of_week=df['Date'].dt.day_name(),
            year_month=df['Date'].dt.to_period('M'),
            transaction_type=np.where(df['amount'] > 0, 'income', 'expense'),
            abs_amount=df['amount'].abs()
        )

        self.df = df
        log.info(f"Cleaned: {after:,} valid rows (removed {before - after:,})")
        log.info(f"Date range: {self.df['Date'].min().date()} → {self.df['Date'].max().date()}")
        log.info(f"Types: {dict(self.df['transaction_type'].value_counts())}")
        return self

    def analyze_by_category(self) -> 'BankTransactionAnalyzer':
        """Analyze expenses by category."""
        log.info("Analyzing spending by category...")
        expenses = self.df[self.df['amount'] < 0].copy()
        if expenses.empty:
            log.warning("No expense transactions found.")
            self.category_summary = pd.DataFrame()
            return self

        total = expenses['abs_amount'].sum()
        summary = (
            expenses.groupby('Category')['abs_amount']
            .agg(total_spent='sum', avg_transaction='mean', count='count')
            .round(2)
            .sort_values('total_spent', ascending=False)
        )
        summary['percentage'] = (summary['total_spent'] / total * 100).round(2)
        self.category_summary = summary

        log.info(f"Total expense: ₹{total:,.2f}")
        log.info("Top categories:\n" + summary.head(5).to_string())
        return self

    def calculate_monthly_balance(self) -> 'BankTransactionAnalyzer':
        """Calculate monthly income, expense, and savings."""
        log.info("Calculating monthly balance...")
        monthly = self.df.groupby('year_month').agg(
            total_income=('amount', lambda x: x[x > 0].sum()),
            total_expenses=('amount', lambda x: x[x < 0].sum()),
            net_balance=('amount', 'sum')
        ).round(2)

        monthly['total_expenses_abs'] = monthly['total_expenses'].abs()
        monthly['savings_rate'] = (
            monthly['net_balance'] / monthly['total_income'].replace(0, np.nan) * 100
        ).round(2)
        monthly['cumulative_balance'] = monthly['net_balance'].cumsum()

        self.monthly_summary = monthly
        log.info("Monthly summary generated")
        return self

    def analyze_spending_patterns(self) -> 'BankTransactionAnalyzer':
        """Analyze spending by day of week."""
        log.info("Analyzing spending patterns...")
        expenses = self.df[self.df['amount'] < 0]
        if expenses.empty:
            log.warning("No expenses for pattern analysis.")
            self.day_spending = pd.DataFrame()
            return self

        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_spending = (
            expenses.groupby('day_of_week')['abs_amount']
            .agg(total='sum', average='mean', count='count')
            .round(2)
            .reindex(day_order, fill_value=0)
        )
        self.day_spending = day_spending
        log.info("Day-of-week spending:\n" + day_spending.to_string())
        return self

    def identify_anomalies(self) -> 'BankTransactionAnalyzer':
        """Detect high-value outliers using Z-score (vectorized, safe)."""
        log.info(f"Detecting anomalies (z > {self.z_threshold})...")
        expenses = self.df[self.df['amount'] < 0].copy()
        if expenses.empty:
            log.warning("No expenses to analyze for anomalies.")
            self.anomalies = pd.DataFrame()
            return self

        # Group and compute mean/std per category
        grouped = expenses.groupby('Category')['abs_amount']
        expenses = expenses.assign(
            mean_cat=grouped.transform('mean'),
            std_cat=grouped.transform('std')
        )

        # Filter categories with std > 0
        valid_mask = expenses['std_cat'] > 0
        if not valid_mask.any():
            log.info("No categories with enough variance for Z-score.")
            self.anomalies = pd.DataFrame()
            return self

        expenses = expenses[valid_mask].copy()
        expenses['z_score'] = (expenses['abs_amount'] - expenses['mean_cat']) / expenses['std_cat']

        # Select anomalies
        anomalies = expenses[expenses['z_score'] > self.z_threshold]
        if anomalies.empty:
            log.info("No anomalies detected.")
            self.anomalies = pd.DataFrame()
            return self

        # Clean up
        anomalies = anomalies.drop(columns=['mean_cat', 'std_cat', 'z_score'])
        anomalies = anomalies.sort_values('abs_amount', ascending=False).reset_index(drop=True)

        self.anomalies = anomalies
        log.warning(f"Found {len(self.anomalies)} anomalous transaction(s)")
        return self

    def identify_high_value(self) -> 'BankTransactionAnalyzer':
        """Identify top X% most expensive transactions."""
        log.info(f"Identifying high-value transactions (>{int(self.high_value_quantile*100)}th percentile)...")
        if self.df is None or self.df.empty:
            self.high_value = pd.DataFrame()
            return self

        threshold = self.df['abs_amount'].quantile(self.high_value_quantile)
        high_value = self.df[self.df['abs_amount'] > threshold]
        self.high_value = high_value.sort_values('abs_amount', ascending=False).reset_index(drop=True)
        log.info(f"Found {len(self.high_value)} high-value transactions (>{threshold:,.2f})")
        return self

    def generate_insights(self) -> 'BankTransactionAnalyzer':
        """Export all analysis results to CSV."""
        log.info("Exporting results...")

        # Always export cleaned data
        path = self.output_dir / "cleaned_transactions.csv"
        self.df.to_csv(path, index=False)
        log.info(f"Cleaned data → {path.name}")

        # Conditional exports
        exports = {
            "category_summary.csv": getattr(self, 'category_summary', None),
            "monthly_summary.csv": getattr(self, 'monthly_summary', None),
            "day_spending.csv": getattr(self, 'day_spending', None),
            "anomalies.csv": getattr(self, 'anomalies', None),
            "high_value_transactions.csv": getattr(self, 'high_value', None),
        }

        for filename, data in exports.items():
            if data is not None and not data.empty:
                path = self.output_dir / filename
                data.to_csv(path, index=False)
                log.info(f"{filename} → {path.name}")

        log.info(f"All outputs saved to: {self.output_dir.resolve()}")
        return self

    def plot_category_pie(self, top_n: int = 8) -> 'BankTransactionAnalyzer':
        """Optionally plot top spending categories (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt

            if not hasattr(self, 'category_summary') or self.category_summary.empty:
                log.warning("No category summary to plot.")
                return self

            top = self.category_summary.head(top_n)
            others_sum = self.category_summary['total_spent'].sum() - top['total_spent'].sum()
            others = pd.Series({'total_spent': others_sum}, name='Others')

            plot_data = top.copy()
            if others_sum > 0:
                plot_data = pd.concat([plot_data, pd.DataFrame([others])])

            plt.figure(figsize=(10, 6))
            plt.pie(plot_data['total_spent'], labels=plot_data.index, autopct='%1.1f%%', startangle=90)
            plt.title(f"Top {top_n} Spending Categories")
            path = self.output_dir / "category_pie_chart.png"
            plt.savefig(path, bbox_inches='tight', dpi=150)
            plt.close()
            log.info(f"Pie chart saved → {path.name}")
        except ImportError:
            log.warning("matplotlib not installed. Install with: pip install matplotlib")
        except Exception as e:
            log.error(f"Plotting failed: {e}")
        return self

    def summary(self) -> Dict[str, Any]:
        """Return a summary dictionary of key metrics."""
        if self.df is None or self.df.empty:
            return {"error": "No data loaded"}

        total_income = self.df[self.df['amount'] > 0]['amount'].sum()
        total_expense = self.df[self.df['amount'] < 0]['amount'].sum()

        return {
            "total_transactions": len(self.df),
            "total_income": round(total_income, 2),
            "total_expense": round(total_expense, 2),
            "net_balance": round(total_income + total_expense, 2),
            "date_range": (
                self.df['Date'].min().date().isoformat(),
                self.df['Date'].max().date().isoformat()
            ),
            "anomalies_count": len(getattr(self, 'anomalies', pd.DataFrame())),
            "high_value_count": len(getattr(self, 'high_value', pd.DataFrame()))
        }


# Entry Point
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("BANK TRANSACTION ANALYZER - CLEAN & PROFESSIONAL")
    print("=" * 60 + "\n")

    analyzer = BankTransactionAnalyzer()
    try:
        (analyzer
         .load_data()
         .clean_and_prepare_data()
         .analyze_by_category()
         .calculate_monthly_balance()
         .analyze_spending_patterns()
         .identify_anomalies()
         .identify_high_value()
         .generate_insights()
         .plot_category_pie())

        print("\nAnalysis completed successfully!")
        print(f"Check outputs in: {analyzer.output_dir.resolve()}")
        print(f"Summary: {analyzer.summary()}\n")
    except Exception as e:
        log.error(f"Analysis failed: {e}")
        raise