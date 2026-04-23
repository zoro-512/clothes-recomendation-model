#!/usr/bin/env python3
"""
Simplified server startup that handles dependency issues gracefully.
"""

import os
import sys
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Set environment variables for SSL and network issues
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['SSL_VERIFY'] = 'FALSE'
os.environ['PYTHONHTTPSVERIFY'] = '0'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_OFFLINE'] = '0'

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

def main():
    try:
        print("🚀 Starting Hybrid AI Recommender Server...")
        print("=" * 50)
        
        # Try to import and start the server
        try:
            import uvicorn
            from src.api.main import app
            
            print("✅ Dependencies loaded successfully")
            print("🌐 Starting server on http://localhost:8006")
            print("📱 Open your browser and navigate to: http://localhost:8006")
            print("🔍 Features available:")
            print("   - Natural language search")
            print("   - Price range filtering")
            print("   - Brand selection")
            print("   - Gender/season filters")
            print("   - Real-time recommendations")
            print()
            print("⚡ Server starting...")
            
            uvicorn.run(
                app,
                host="0.0.0.0",
                port=8006,
                reload=False,  # Disable reload to avoid issues
                log_level="info"
            )
            
        except ImportError as e:
            print(f"❌ Import error: {e}")
            print("📦 Installing missing dependencies...")
            os.system("pip install uvicorn fastapi")
            print("🔄 Retrying startup...")
            
        except Exception as e:
            print(f"❌ Server startup error: {e}")
            print("🔧 Trying alternative startup method...")
            
            # Fallback: try to run with minimal dependencies
            try:
                os.system("python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8006 --log-level info")
            except Exception as e2:
                print(f"❌ Fallback also failed: {e2}")
                return False
                
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
        return True
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("✅ Server startup completed")
    else:
        print("❌ Server startup failed")
        sys.exit(1)
