#!/usr/bin/env python3
# test_mode_with_args.py - Example of using the enhanced mode endpoint with arguments
import requests
import json

API_BASE = "http://127.0.0.1:7070/api"

def test_mode_endpoint():
    """Test the enhanced mode endpoint with different argument combinations"""
    
    print("Testing Enhanced Loadpoint Mode Endpoint")
    print("=" * 50)
    
    # Test 1: Basic mode change (no arguments)
    print("\n1. Basic mode change (no arguments):")
    url = f"{API_BASE}/loadpoints/1/mode/pv"
    response = requests.post(url)
    print(f"POST {url}")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # Test 2: Mode change with priority argument
    print("\n2. Mode change with priority argument:")
    url = f"{API_BASE}/loadpoints/1/mode/now"
    data = {
        "priority": 5
    }
    response = requests.post(url, json=data)
    print(f"POST {url}")
    print(f"Body: {json.dumps(data, indent=2)}")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # Test 3: Mode change with multiple arguments
    print("\n3. Mode change with multiple arguments:")
    url = f"{API_BASE}/loadpoints/1/mode/pv"
    data = {
        "priority": 3,
        "enable_threshold": 1500.0,  # 1.5 kW
        "disable_threshold": 500.0   # 0.5 kW
    }
    response = requests.post(url, json=data)
    print(f"POST {url}")
    print(f"Body: {json.dumps(data, indent=2)}")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # Test 4: Error case - invalid priority
    print("\n4. Error case - invalid priority:")
    url = f"{API_BASE}/loadpoints/1/mode/pv"
    data = {
        "priority": -1  # Invalid: negative priority
    }
    response = requests.post(url, json=data)
    print(f"POST {url}")
    print(f"Body: {json.dumps(data, indent=2)}")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # Test 5: Error case - invalid threshold type
    print("\n5. Error case - invalid threshold type:")
    url = f"{API_BASE}/loadpoints/1/mode/pv"
    data = {
        "enable_threshold": "not_a_number"
    }
    response = requests.post(url, json=data)
    print(f"POST {url}")
    print(f"Body: {json.dumps(data, indent=2)}")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # Test 6: Get current state to see the changes
    print("\n6. Get current loadpoint state:")
    url = f"{API_BASE}/state?jq=.loadpoints[0]"
    response = requests.get(url)
    print(f"GET {url}")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print(f"Current Loadpoint State: {json.dumps(response.json(), indent=2)}")

if __name__ == "__main__":
    try:
        test_mode_endpoint()
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to API server.")
        print("Make sure the HEMS API server is running on port 7070")
        print("Start it with: python hems_with_EVCC.py --api --port 7070")
    except Exception as e:
        print(f"ERROR: {e}")