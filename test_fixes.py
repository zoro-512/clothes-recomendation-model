import requests
import json

def test_jeans_search():
    """Test that 'jeans' search returns relevant products."""
    try:
        payload = {
            "query": "jeans",
            "user_id": "USER_001",
            "top_k": 5,
            "season": "unknown",
            "gender": "all"
        }
        
        response = requests.post("http://localhost:8003/recommend", 
                               json=payload,
                               headers={"Content-Type": "application/json"})
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Jeans search working!")
            print(f"Query: {data['query']}")
            print(f"Total candidates evaluated: {data['total_candidates_evaluated']}")
            print(f"Number of recommendations: {len(data['recommendations'])}")
            
            # Check if results are relevant to jeans
            jeans_count = 0
            for i, rec in enumerate(data['recommendations']):
                product_name = rec['product_name'].lower()
                product_type = rec.get('product_type', '').lower()
                product_group = rec.get('product_group', '').lower()
                description = rec.get('description', '').lower()
                
                is_jeans_related = ('jean' in product_name or 
                                  'jean' in product_type or 
                                  'jean' in product_group or 
                                  'jean' in description)
                
                if is_jeans_related:
                    jeans_count += 1
                
                print(f"  {i+1}. {rec['product_name']}")
                print(f"     Type: {rec.get('product_type', 'N/A')}")
                print(f"     Group: {rec.get('product_group', 'N/A')}")
                print(f"     Jeans-related: {'Yes' if is_jeans_related else 'No'}")
                print()
            
            print(f"Jeans-related results: {jeans_count}/{len(data['recommendations'])}")
            
            if jeans_count >= len(data['recommendations']) * 0.6:  # At least 60% relevant
                print("✅ Good relevance for jeans search!")
                return True
            else:
                print("⚠️  Low relevance for jeans search")
                return False
                
        else:
            print(f"❌ Jeans search failed: {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing jeans search: {e}")
        return False

def test_server_health():
    """Test if server is running and has embedding model."""
    try:
        response = requests.get("http://localhost:8003/health")
        if response.status_code == 200:
            health = response.json()
            print(f"✅ Server health check passed!")
            print(f"Status: {health['status']}")
            print(f"LLM Mode: {health['llm_mode']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Server not accessible: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Fixes for HuggingFace Connection and Jeans Search")
    print("=" * 70)
    
    # Test server health
    health_ok = test_server_health()
    print()
    
    # Test jeans search
    jeans_ok = test_jeans_search()
    print()
    
    if health_ok and jeans_ok:
        print("🎉 All fixes working correctly!")
    else:
        print("⚠️  Some issues remain. Please check the implementation.")
