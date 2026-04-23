#!/usr/bin/env python3
"""
Test script to verify Myntra data loading works correctly.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

def test_myntra_data():
    """Test loading real Myntra data."""
    try:
        print("🧪 Testing Real Myntra Data Loading")
        print("=" * 50)
        
        # Import required modules
        from src.data_pipeline.preprocess import load_myntra_data
        import pandas as pd
        
        print("✅ Imports successful")
        
        # Load real Myntra data
        print("📦 Loading real Myntra dataset...")
        articles = load_myntra_data()
        
        print(f"✅ Loaded {len(articles)} real products from Myntra dataset")
        print(f"📊 Dataset shape: {articles.shape}")
        print(f"📋 Columns: {list(articles.columns)}")
        
        # Show sample of real data
        print("\n👕 Sample REAL Myntra Products:")
        print("-" * 50)
        for i in range(5):
            row = articles.iloc[i]
            print(f"{i+1}. {row.get('prod_name', 'N/A')}")
            print(f"   Brand: {row.get('brand_name', 'N/A')}")
            print(f"   Price: ₹{row.get('price', 'N/A')}")
            print(f"   Type: {row.get('product_type_name', 'N/A')}")
            print(f"   URL: {row.get('product_url', 'N/A')[:50]}...")
            print()
        
        # Test filtering
        print("🔍 Testing Price Filtering on Real Data:")
        price_filtered = articles[articles['price'] >= 500]
        print(f"Products ≥ ₹500: {len(price_filtered)}")
        
        print("\n🏷️  Testing Brand Filtering on Real Data:")
        brands = articles['brand_name'].value_counts().head(5)
        print("Top 5 brands:")
        for brand, count in brands.items():
            print(f"  {brand}: {count} products")
        
        print("\n👖 Testing 'Jeans' Search on Real Data:")
        jeans_mask = (
            articles['product_type_name'].str.contains('jean', case=False, na=False) |
            articles['prod_name'].str.contains('jean', case=False, na=False) |
            articles['detail_desc'].str.contains('jean', case=False, na=False)
        )
        jeans_products = articles[jeans_mask]
        print(f"Found {len(jeans_products)} jeans products")
        
        if len(jeans_products) > 0:
            print("Sample jeans products:")
            for i in range(min(3, len(jeans_products))):
                row = jeans_products.iloc[i]
                print(f"  - {row.get('prod_name', 'N/A')} ({row.get('brand_name', 'N/A')}) - ₹{row.get('price', 'N/A')}")
        
        print("\n✅ All tests passed! Real Myntra data is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Error testing Myntra data: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_myntra_data()
    if success:
        print("\n🎉 Real Myntra data is ready for use!")
    else:
        print("\n⚠️  Issues found with Myntra data loading.")
