import requests
import xml.etree.ElementTree as ET
import json
import os
from dotenv import load_dotenv

load_dotenv()

route_url = "http://ws.bus.go.kr/api/rest/busRouteInfo/getStaionByRoute"

route_params = {
    "busRouteId": 100100033,
    "serviceKey": os.getenv("BUS_API_KEY")
}


def fetch_route_data():
    response = requests.get(route_url, params=route_params)

    root = ET.fromstring(response.text)

    stations = []
    total_dist = 0

    for item in root.findall(".//itemList"):
        seq = int(item.findtext("seq"))
        name = item.findtext("stationNm")
        lat = float(item.findtext("gpsY"))
        lon = float(item.findtext("gpsX"))
        dist = int(item.findtext("fullSectDist") or 0)
        speed = int(item.findtext("sectSpd") or 0)
        direction = item.findtext("direction")
        transYn = item.findtext("transYn")

        total_dist += dist

        stations.append({
            "seq": seq,
            "name": name,
            "lat": lat,
            "lon": lon,
            "dist": dist,
            "speed": speed,
            "cum_dist": total_dist,
            "direction": direction,
            "is_turn": transYn == "Y"
        })

    stations.sort(key=lambda x: x["seq"])

    seq_list = [s["seq"] for s in stations]
    seq_min = min(seq_list)
    seq_max = max(seq_list)

    return stations, seq_min, seq_max

def save_to_json(stations, seq_min, seq_max, filename="route_160.json"):
    data = {
        "meta": {
            "seq_min": seq_min,
            "seq_max": seq_max,
            "stations_count": len(stations)
        },
        "stations": stations
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_from_json(filename="route_160.json"):
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    meta = data["meta"]
    stations = data["stations"]

    return stations, meta


if __name__ == "__main__":
    if not route_params["serviceKey"]:
        print("[ERROR] .env 파일에 BUS_API_KEY가 설정되지 않았습니다.")
        exit(1)

    print("저장 시작")
    stations, seq_min, seq_max = fetch_route_data()
    print("정류소 개수:", len(stations))
    save_to_json(stations, seq_min, seq_max)
    print("JSON 저장 완료")