# Bank Transaction Analyzer  

> **A clean, production-ready personal bank transaction analyzer in Python.**  
> Auto-detects CSV, cleans data, detects anomalies, and exports insights.

---

## Features

- **Auto-detects** `transactions.csv` in `data/` or root  
- **Smart parsing** of date, amount, category, and type  
- **Full method chaining** for clean pipeline  
- **Professional logging** with timestamps  
- **Anomaly detection** using Z-score  
- **Monthly, category, and weekly spending analysis**  
- **CSV exports + optional pie chart** (matplotlib)  
- **Fully configurable** (thresholds, paths, date format)

---

## Project Structure

bank-transaction-analyzer/
├── data/
│   └── transactions.csv          # Your bank export
├── outputs/                      # Generated reports & charts
├── main.py              # Main script
├── README.md

---

## Usage

1. Prepare Your CSV
Example format (column names can vary):
csv
Date,Amount,Category,Income/Expense
15/03/2025,1500,Food,Expense
16/03/2025,50000,Salary,Income

2. Place CSV in:

data/transactions.csv (recommended)
or transactions.csv in project root

3. Run:
python main.py

---

# Example Console Output
============================================================
BANK ANALYZER - FINAL VERSION
============================================================

14:22:01 | INFO     | Data: data/transactions.csv
14:22:01 | INFO     | Loaded 1,284 rows
...
14:22:05 | WARNING  | Found 3 anomalous transaction(s)

SUCCESS!
Outputs: /path/to/outputs
Summary: {'total': 1240, 'income': 285000.0, 'expense': -182400.0, 'net': 102600.0}