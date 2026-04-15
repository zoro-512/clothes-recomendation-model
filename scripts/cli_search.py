import sys
import json
import requests

API_URL = "http://127.0.0.1:8000"

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def check_health():
    try:
        requests.get(f"{API_URL}/health", timeout=2)
        return True
    except requests.exceptions.RequestException:
        return False

def interactive_cli():
    print(Colors.HEADER + Colors.BOLD + "="*50)
    print("Hybrid E-commerce Recommender")
    print("="*50 + Colors.ENDC)
    
    if not check_health():
        print(Colors.FAIL + " API Server is NOT running!" + Colors.ENDC)
        print(Colors.WARNING + "Please start the server first by running in a separate terminal:" + Colors.ENDC)
        print("C:\\venvs\\recom\\Scripts\\python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000\n")
        sys.exit(1)

    print(Colors.GREEN + " API Server Connected!" + Colors.ENDC)
    print("Type your product requests in natural language. Type 'exit' to quit.\n")

    while True:
        try:
            query = input(Colors.BOLD + Colors.CYAN + "Search> " + Colors.ENDC).strip()
            
            if not query:
                continue
            if query.lower() in ('exit', 'quit', 'q'):
                print(Colors.GREEN + "Goodbye!" + Colors.ENDC)
                break

            print(Colors.WARNING + "   Thinking..." + Colors.ENDC)
            response = requests.post(
                f"{API_URL}/recommend",
                json={
                    "query": query,
                    "user_id": "CLI_USER",
                    "top_k": 3,
                    "season": "unknown"
                }
            )
            
            if not response.ok:
                print(Colors.FAIL + f"Error: {response.text}" + Colors.ENDC)
                continue
                
            data = response.json()
            recs = data.get("recommendations", [])
            
            if not recs:
                print(Colors.WARNING + "No products matched your search." + Colors.ENDC)
                continue

            print(f"\n{Colors.BOLD} Top {len(recs)} Recommendations:{Colors.ENDC}")
            for idx, item in enumerate(recs, 1):
                match_pct = round(item.get('semantic_score', 0) * 100)
                
                print(f"\n{Colors.BLUE}[Rank {idx}] {Colors.BOLD}{item['product_name']}{Colors.ENDC} ({item['product_group']})")
                print(f"       {Colors.GREEN}► Semantic Match: {match_pct}%{Colors.ENDC}")
                print(f"       {Colors.ENDC}► Desc: {item['description']}{Colors.ENDC}")
                
                if item.get("explanation"):
                    print(f"       {Colors.HEADER}► AI Stylist: {item['explanation']}{Colors.ENDC}")
            
            print("\n" + "-"*50 + "\n")

        except KeyboardInterrupt:
            print("\n" + Colors.GREEN + "Goodbye!" + Colors.ENDC)
            break
        except Exception as e:
            print(Colors.FAIL + f"\nSomething went wrong: {e}" + Colors.ENDC)

if __name__ == "__main__":
    interactive_cli()
