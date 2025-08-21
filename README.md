# nifty50-valuation-framework
Python-based valuation framework for NIFTY 50 stocks integrating P/E ratio and CAPM to detect mispricing, map performance against valuation, and visualize results with quadrant-style outputs.

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](#)  
[![pandas](https://img.shields.io/badge/Library-pandas-yellow)](#)  
[![NumPy](https://img.shields.io/badge/Library-NumPy-lightgrey)](#)  
[![yfinance](https://img.shields.io/badge/API-yfinance-lightblue)](#)  
[![statsmodels](https://img.shields.io/badge/Modeling-statsmodels-green)](#)  
[![matplotlib](https://img.shields.io/badge/Visualization-matplotlib-orange)](#)  
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)  

**Tagline:**  
A data-driven framework to uncover undervalued outperformers and overvalued underperformers in the NIFTY 50.

---

## Overview
This repository implements an end-to-end pipeline to:
- Pull NIFTY 50 constituents and fundamentals,
- Compute industry-level P/E statistics,
- Classify stocks with two valuation methods (Thumb-rule and Industry-range),
- Validate labels against recent price performance,
- Apply CAPM to overvalued names,
- Optionally forecast prices for selected stocks.

The approach emphasizes reproducibility and clear, auditable intermediate outputs.

---

## Data & Sources
- **Constituents:** NSE index list (CSV)  
- **Market & Fundamentals:** Yahoo Finance via `yfinance` (P/E, beta, prices)  
- **Historical Market Returns:** Local CSV (`nifty50_historical_return.csv`)

> The notebook automates symbol retrieval from the NSE list, then uses `yfinance` for trailing P/E, beta, and price history. Intermediate CSVs are written at each step for traceability.

---

## Methodology

1. **Universe & Fundamentals**
   - Fetch NIFTY 50 tickers, append “.NS” for Yahoo compatibility.
   - Collect trailing P/E for each constituent and write `nifty50_pe_ratios.csv`.

2. **Industry Aggregation**
   - For each stock, fetch industry from `yfinance`.
   - Compute industry P/E stats (Avg, Min, Max, Std Dev, Count) → `nifty50_industry_pe_analysis.csv`.
   - Materialize stock–industry pairs → `nifty50_stocks_by_industry.csv`.

3. **Valuation Labels**
   - **Thumb-rule:**  
     - P/E < 10 → Undervalued  
     - 10 ≤ P/E ≤ 40 → Fairly Valued  
     - P/E > 50 → Overvalued  
     Saves `nifty50_pe_valuation_thumbrule.csv`.
   - **Industry-range:**  
     - Uses industry Avg ± Std Dev as a bounded range (lower bound floored at 0; upper bound capped at 2×Avg).  
     Saves `nifty50_stocks_valuation_range_analysis.csv`.

4. **Performance Check (3-Month)**
   - Pull 3-month total price change from Yahoo for each symbol.
   - Compare average 3-month returns by label for both methods and save merged dataset to `nifty50_valuation_performance.csv`.
   - Create a **Final Valuation**: if both methods agree, take that label; otherwise default to “Overvalued”. Output `nifty50_final_stock_valuation.csv`.

5. **CAPM on Overvalued Names**
   - Market return \(R_m\) from the average “Annual” column in `nifty50_historical_return.csv`; risk-free rate \(R_f = 6.58\%\).
   - Retrieve stock beta; where missing, fall back to industry-average beta; if unavailable, use 1.0.
   - Compute expected return \(R_e = R_f + \beta(R_m - R_f)\).
   - Compare \(R_e\) with 3-month actual return; label as **Matched CAPM** (±1%), **Underperformed**, or **Outperformed**.  
   Saves `overvalued_stocks_capm.csv` and `capm_vs_price_change.csv`, and a tidy join `overvalued_stocks_final_performance.csv`.

6. **Optional Forecasting**
   - For overvalued and underperforming stocks, fit ARIMA(1,1,1) models to the last ~2 years of daily closes and produce 90-day forecasts and plots.

---

## Key Results (from the included run)
- Estimated market return \(R_m \approx 15.33\%\) (simple average of historical “Annual” returns).  
- Example label performance (mean 3-month change):  
  - **Thumb-rule:** Overvalued ≈ 8.74%, Fairly Valued ≈ 3.02%, Undervalued ≈ −0.98%  
  - **Industry-range:** Fairly Valued ≈ 6.22%, Overvalued ≈ 0.42%, Undervalued ≈ −9.73%  
- CAPM vs 3-month change annotated per stock with “Matched CAPM / Underperformed / Outperformed”.

> Exact numbers are from the saved run in the notebook and depend on fetch date, ticker coverage, and data availability.

---
