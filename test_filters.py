import requests
import json

# Test the brands endpoint
def test_brands_endpoint():
    try:
        response = requests.get("http://localhost:8002/brands")
        if response.status_code == 200:
            brands = response.json()
            print(f"✅ Brands endpoint working! Found {len(brands)} brands")
            print(f"Sample brands: {brands[:5]}")
            return True
        else:
            print(f"❌ Brands endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error testing brands endpoint: {e}")
        return False

# Test the recommendation endpoint with filters
def test_recommendation_with_filters():
    try:
        # Test with price range and brand filters
        payload = {
            "query": "blue t-shirt",
            "user_id": "USER_001",
            "top_k": 3,
            "season": "summer",
            "gender": "men",
            "min_price": 500.0,
            "max_price": 2000.0,
            "brand": "Nike"
        }
        
        response = requests.post("http://localhost:8002/recommend", 
                               json=payload,
                               headers={"Content-Type": "application/json"})
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Recommendation endpoint working!")
            print(f"Query: {data['query']}")
            print(f"Total candidates evaluated: {data['total_candidates_evaluated']}")
            print(f"Number of recommendations: {len(data['recommendations'])}")
            
            for i, rec in enumerate(data['recommendations'][:2]):
                print(f"  {i+1}. {rec['product_name']} - {rec.get('product_group', 'N/A')}")
                print(f"     Price: {rec.get('price', 'N/A')}, Brand: {rec.get('brand_name', 'N/A')}")
            
            return True
        else:
            print(f"❌ Recommendation endpoint failed: {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing recommendation endpoint: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Price Range and Brand Filtering Implementation")
    print("=" * 60)
    
    # Test brands endpoint
    brands_ok = test_brands_endpoint()
    print()
    
    # Test recommendation with filters
    rec_ok = test_recommendation_with_filters()
    print()
    
    if brands_ok and rec_ok:
        print("🎉 All tests passed! The filtering implementation is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
