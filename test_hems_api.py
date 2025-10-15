#!/usr/bin/env python3
# test_hems_api.py - Test script for HEMS EVCC API
import requests
import json
import time

API_BASE = "http://127.0.0.1:7070/api"

def test_endpoint(method, endpoint, data=None, description=""):
    """Test an API endpoint"""
    url = f"{API_BASE}{endpoint}"
    print(f"\n{description}")
    print(f"{method} {url}")
    
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            if data:
                response = requests.post(url, json=data)
            else:
                response = requests.post(url)
        elif method == "DELETE":
            response = requests.delete(url)
        
        print(f"Status: {response.status_code}")
        if response.headers.get('content-type', '').startswith('application/json'):
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2)}")
        else:
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to API server. Make sure it's running.")
    except Exception as e:
        print(f"ERROR: {e}")

def main():
    print("HEMS EVCC API Test Script")
    print("=" * 40)
    
    # Test basic endpoints
    test_endpoint("GET", "/health", description="1. Health Check")
    
    test_endpoint("GET", "/state", description="2. Get System State")
    
    # Test loadpoint endpoints
    test_endpoint("POST", "/loadpoints/1/mode/pv", description="3. Set Loadpoint Mode to PV")
    
    test_endpoint("POST", "/loadpoints/1/phases/3", description="4. Set Loadpoint to 3 phases")
    
    test_endpoint("POST", "/loadpoints/1/mincurrent/6.0", description="5. Set Min Current to 6A")
    
    test_endpoint("POST", "/loadpoints/1/maxcurrent/16.0", description="6. Set Max Current to 16A")
    
    # Test vehicle endpoints
    test_endpoint("POST", "/vehicles/ev1/minsoc/20.0", description="7. Set Vehicle Min SoC to 20%")
    
    test_endpoint("POST", "/vehicles/ev1/limitsoc/80.0", description="8. Set Vehicle Limit SoC to 80%")
    
    test_endpoint("POST", "/loadpoints/1/vehicle/ev1", description="9. Assign Vehicle to Loadpoint")
    
    # Test battery endpoints
    test_endpoint("POST", "/batterymode/hold", description="10. Set Battery Mode to Hold")
    
    test_endpoint("POST", "/buffersoc/10.0", description="11. Set Battery Buffer SoC to 10%")
    
    test_endpoint("POST", "/prioritysoc/30.0", description="12. Set Battery Priority SoC to 30%")
    
    # Test HEMS-specific endpoints
    measurements_data = {
        "base_load_kw": 2.5,
        "solar_kw": 4.0,
        "house_temp_c": 19.0,
        "battery_energy_kwh": 12.0,
        "people_presence_pct": 80.0,
        "desired_charger_kw": 11.0,
        "needs_heating": True,
        "step_hours": 1.0
    }
    test_endpoint("POST", "/hems/measurements", data=measurements_data, 
                 description="13. Update HEMS Measurements")
    
    test_endpoint("GET", "/hems/rules", description="14. Get Rules and Metarules")
    
    # Get updated state
    test_endpoint("GET", "/state", description="15. Get Updated System State")
    
    # Reset battery mode
    test_endpoint("DELETE", "/batterymode", description="16. Reset Battery Mode")
    
    print(f"\n{'='*40}")
    print("API Test Complete!")
    print("The HEMS system now has an EVCC-compatible REST API that can be integrated")
    print("with external systems for automated control and monitoring.")

if __name__ == "__main__":
    main()