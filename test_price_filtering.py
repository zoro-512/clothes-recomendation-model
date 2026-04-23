#!/usr/bin/env python3
"""
Test script to verify price filtering works correctly on real Myntra data.
"""

import requests
import json

def test_price_filtering():
    """Test price range filtering with real Myntra products."""
    
    base_url = "http://localhost:8007"
    
    print("🧪 Testing Price Range Filtering on Real Myntra Data")
    print("=" * 60)
    
    # Test 1: Search for jeans without price filter
    print("📦 Test 1: Search 'jeans' without price filter")
    try:
        payload = {
            "query": "jeans",
            "user_id": "USER_001",
            "top_k": 5,
            "season": "unknown",
            "gender": "all"
        }
        
        response = requests.post(f"{base_url}/recommend", json=payload)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Found {len(data['recommendations'])} jeans products")
            for i, rec in enumerate(data['recommendations']):
                print(f"  {i+1}. {rec['product_name']} - ₹{rec.get('price', 'N/A')}")
        else:
            print(f"❌ Error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    print()
    
    # Test 2: Search for jeans with price range 500-1000
    print("💰 Test 2: Search 'jeans' with price range ₹500-₹1000")
    try:
        payload = {
            "query": "jeans",
            "user_id": "USER_001",
            "top_k": 5,
            "season": "unknown",
            "gender": "all",
            "min_price": 500.0,
            "max_price": 1000.0
        }
        
        response = requests.post(f"{base_url}/recommend", json=payload)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Found {len(data['recommendations'])} jeans products in price range")
            
            all_in_range = True
            for i, rec in enumerate(data['recommendations']):
                price = rec.get('price')
                if price is not None:
                    price_in_range = 500 <= price <= 1000
                    if not price_in_range:
                        all_in_range = False
                    print(f"  {i+1}. {rec['product_name']} - ₹{price} {'✅' if price_in_range else '❌'}")
                else:
                    print(f"  {i+1}. {rec['product_name']} - Price N/A")
            
            if all_in_range:
                print("✅ ALL products are within the specified price range!")
            else:
                print("❌ Some products are outside the specified price range!")
                return False
        else:
            print(f"❌ Error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    print()
    
    # Test 3: Search for jeans with price range 1000-2000
    print("💰 Test 3: Search 'jeans' with price range ₹1000-₹2000")
    try:
        payload = {
            "query": "jeans",
            "user_id": "USER_001",
            "top_k": 5,
            "season": "unknown",
            "gender": "all",
            "min_price": 1000.0,
            "max_price": 2000.0
        }
        
        response = requests.post(f"{base_url}/recommend", json=payload)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Found {len(data['recommendations'])} jeans products in price range")
            
            all_in_range = True
            for i, rec in enumerate(data['recommendations']):
                price = rec.get('price')
                if price is not None:
                    price_in_range = 1000 <= price <= 2000
                    if not price_in_range:
                        all_in_range = False
                    print(f"  {i+1}. {rec['product_name']} - ₹{price} {'✅' if price_in_range else '❌'}")
                else:
                    print(f"  {i+1}. {rec['product_name']} - Price N/A")
            
            if all_in_range:
                print("✅ ALL products are within the specified price range!")
            else:
                print("❌ Some products are outside the specified price range!")
                return False
        else:
            print(f"❌ Error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    print()
    
    # Test 4: Search for shirts with brand filter and price range
    print("🏷️  Test 4: Search 'shirts' with brand 'Roadster' and price range ₹800-₹1500")
    try:
        payload = {
            "query": "shirts",
            "user_id": "USER_001",
            "top_k": 5,
            "season": "unknown",
            "gender": "all",
            "min_price": 800.0,
            "max_price": 1500.0,
            "brand": "Roadster"
        }
        
        response = requests.post(f"{base_url}/recommend", json=payload)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Found {len(data['recommendations'])} Roadster shirts in price range")
            
            all_correct = True
            for i, rec in enumerate(data['recommendations']):
                name = rec['product_name']
                price = rec.get('price')
                brand_correct = 'Roadster' in name or 'roadster' in name.lower()
                price_correct = price is not None and 800 <= price <= 1500
                
                if not brand_correct or not price_correct:
                    all_correct = False
                
                brand_status = "✅" if brand_correct else "❌"
                price_status = "✅" if price_correct else "❌"
                print(f"  {i+1}. {name} - ₹{price} [Brand: {brand_status}] [Price: {price_status}]")
            
            if all_correct:
                print("✅ ALL products match the brand and price criteria!")
            else:
                print("❌ Some products don't match the brand or price criteria!")
                return False
        else:
            print(f"❌ Error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    print()
    print("🎉 All price filtering tests passed!")
    return True

if __name__ == "__main__":
    success = test_price_filtering()
    if success:
        print("\n✅ Price filtering is working correctly on real Myntra data!")
        print("\n📱 You can now access the application at: http://localhost:8007")
        print("🔍 Try searching for products with different price ranges!")
    else:
        print("\n❌ Price filtering tests failed. Check the server logs.")
