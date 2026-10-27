import urllib.request
import json

def check():
    # 1. Health
    with urllib.request.urlopen('http://127.0.0.1:8000/health') as r:
        print('Health:', r.read().decode())

    # 2. Safe zones
    with urllib.request.urlopen('http://127.0.0.1:8000/api/safe-zones/sim_01') as r:
        zones = json.loads(r.read().decode())
        print(f"Safe Zones for sim_01 ({len(zones)}):", zones[0]['name'], f"Radius: {zones[0]['radius_m']}m")

    # 3. Wearable
    with urllib.request.urlopen('http://127.0.0.1:8000/api/wearable/sim_01/latest') as r:
        w = json.loads(r.read().decode())
        print(f"Wearable for sim_01: {w['status']}, {w['reading']['heart_rate_bpm']} BPM ({w['reading']['sensor_status']})")

    # 4. Risk history
    with urllib.request.urlopen('http://127.0.0.1:8000/api/risk-history/sim_01?limit=3') as r:
        rh = json.loads(r.read().decode())
        print(f"Risk history for sim_01: {len(rh)} points, latest score: {rh[-1]['risk_score']}, tier: {rh[-1]['risk_tier']}")

    # 5. Alerts
    with urllib.request.urlopen('http://127.0.0.1:8000/api/alerts/user/sim_01') as r:
        alerts = json.loads(r.read().decode())
        print(f"Alerts for sim_01: {len(alerts)} records")

if __name__ == '__main__':
    check()
