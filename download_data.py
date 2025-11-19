"""
Download Market Data
====================

This script downloads real market data from Yahoo Finance

Run: python download_data.py
"""
import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime

def download_data():
    """Download all market data"""
    
    print("="*70)
    print("DOWNLOADING REAL MARKET DATA FROM YAHOO FINANCE")
    print("="*70)
    print()
    
    # Create data directory if it doesn't exist
    data_dir = Path('data/raw')
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created directory: {data_dir}")
    print()
    
    # Define what to download
    tickers = {
        'SPY': 'S&P 500 ETF',
        'QQQ': 'NASDAQ-100 ETF',
        'AAPL': 'Apple Inc.',
        'TSLA': 'Tesla Inc.',
        '^VIX': 'Volatility Index'
    }
    
    # Date range
    start_date = '2010-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"Downloading data from {start_date} to {end_date}")
    print()
    
    # Download each ticker
    success_count = 0
    
    for ticker, description in tickers.items():
        print(f"📊 Downloading {ticker} ({description})...")
        
        try:
            # Download from Yahoo Finance
            data = yf.download(
                ticker, 
                start=start_date, 
                end=end_date,
                progress=False
            )
            
            if len(data) > 0:
                # Save to CSV
                # Remove ^ from filename (VIX -> VIX.csv, not ^VIX.csv)
                filename = ticker.replace('^', '') + '.csv'
                filepath = data_dir / filename
                
                data.to_csv(filepath)
                
                print(f"   ✓ Downloaded {len(data)} days of data")
                print(f"   ✓ Saved to: {filepath}")
                print(f"   Date range: {data.index[0].date()} to {data.index[-1].date()}")
                print()
                
                success_count += 1
            else:
                print(f"   ✗ No data returned for {ticker}")
                print()
                
        except Exception as e:
            print(f"   ✗ Error downloading {ticker}: {e}")
            print()
    
    # Summary
    print("="*70)
    print(f"DOWNLOAD COMPLETE: {success_count}/{len(tickers)} successful")
    print("="*70)
    print()
    
    # List downloaded files
    print("📁 Files in data/raw/:")
    csv_files = list(data_dir.glob('*.csv'))
    
    if csv_files:
        for csv_file in sorted(csv_files):
            # Get file size
            size_mb = csv_file.stat().st_size / (1024 * 1024)
            print(f"   - {csv_file.name} ({size_mb:.2f} MB)")
        print()
        print("✅ Data download successful!")
        print()
        print("Next steps:")
        print("   1. Run: python check_real_data.py (to verify)")
        print("   2. Continue to Day 2: Feature Engineering")
    else:
        print("   ✗ No files downloaded")
        print()
        print("Troubleshooting:")
        print("   - Check internet connection")
        print("   - Try: pip install --upgrade yfinance")
        print("   - Run script again")

if __name__ == "__main__":
    # Check if yfinance is installed
    try:
        import yfinance as yf
        download_data()
    except ImportError:
        print("❌ yfinance not installed!")
        print()
        print("Install it with:")
        print("   pip install yfinance")
        print()
        print("Then run this script again.")