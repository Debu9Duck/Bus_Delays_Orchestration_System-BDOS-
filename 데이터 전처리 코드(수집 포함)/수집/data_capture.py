import requests
import xml.etree.ElementTree as ET
import time
from datetime import datetime
import csv
import os
import json
from bus_route import load_from_json
from dotenv import load_dotenv

load_dotenv()  # 로컬 .env 로드 (Render 환경변수와 자동 호환)

SERVICE_KEY = os.environ["SERVICE_KEY"]

def clear_all(): #예상치 못한 중지일 시 해당함수 사용
    bus_state.clear()
    last_arrival_time.clear()
    last_processed_tm.clear()
    session_dicts.clear()

def clear_vehicle(vehId):
    bus_state.pop(vehId, None)
    last_arrival_time.pop(vehId, None)
    last_processed_tm.pop(vehId, None)
    session_dicts.pop(vehId, None)

def finalize_vehicle(vehId, sectOrd, dataTm, is_jump, jsonl_file):
    #운행 종료 시 진행중인 세션 저장 후 상태 정리
    session_dict = session_dicts.get(vehId)

    if session_dict and session_dict.get("10secDist"):
        session_dict["endtime"] = dataTm
        session_dict["toSect"] = sectOrd

        record = {
            "routeId": location_params["busRouteId"],
            "seq": session_dict["fromSect"],
            "vehId": vehId,
            "toSect": sectOrd,
            "starttime": session_dict["starttime"],
            "endtime": dataTm,
            "is_jump": is_jump,
            "10secDist": session_dict["10secDist"],
            "10secTm": session_dict["10secTm"]
        }
        jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        jsonl_file.flush()
        print(f"[운행종료] {vehId} 마지막 세션 기록 완료")
    




location_url = "http://ws.bus.go.kr/api/rest/buspos/getBusPosByRtid"

stations, meta = load_from_json()

location_params = {
    "busRouteId": 100100033,
    "serviceKey": SERVICE_KEY
}

bus_state = {} #버스 상태 추적
last_arrival_time = {} #이전 정류장 버스 도착 시간
session_dicts = {} #버스별 구간 이동 세션
last_api_time = None #api동기화용 변수 1 폐기
last_processed_tm = {} #api동기화용 변수 2

jump_count = 0

filename = r"D:\세세한네비게이션프로젝트\bus_data.csv" #r은 \b랑 \세가 에러코드 안뜨게 하는거
file_exists = os.path.isfile(filename)


with open(filename, "a", newline="", encoding="UTF-8") as csvfile:

    stations, meta = load_from_json("route_160.json") #받는쪽은 일치해야 한다
    station_info = {s["seq"]: s for s in stations}
    turn_map = {s["seq"]: s["is_turn"] for s in stations}

    jsonl_file = open(f"{location_params['busRouteId']}.jsonl", "a", encoding="utf-8")


    fieldnames = [
        "vehId", 
        "plainNo", 

        "fromSect", 
        "toSect",

        "starttime",
        "endtime", 

        "prevSectDist",
        "currSectDist", 

        "trip_id",
        "trip_direction",

        "seq",
        "stationName",

        "travelTimeSec"
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    #헤더 작성, 처음 만들 떄만 해당
    if not file_exists:
        writer.writeheader()

    try:
        skip_record = False
        


        while True:
            now = datetime.now()

            #n시 이후 종료
            if now.hour >= 24:
                print("수집 종료")
                break

            #api 호출, 예외처리
            try:
                location_response = requests.get(location_url, params=location_params, timeout=5)
                location_response.raise_for_status()
            
            except requests.exceptions.Timeout:
                print("[ERROR] 타임아웃 발생, 재시도")
                time.sleep(10)
                continue

            except requests.exceptions.RequestException as e:
                print(f"[ERROR] 요청 실패: {e}")

            #XML 파싱
            try:
                location_root = ET.fromstring(location_response.text)
            except ET.ParseError:
                print("[ERROR] XML 파싱 실패")
                time.sleep(10)
                continue

            #api 에러코드 체크 확장 가능
            headerCd = location_root.findtext(".//headerCd")
            headerMsg = location_root.findtext(".//headerMsg")

            if headerCd != "0":
                print(f"[API ERROR] code={headerCd}, msg={headerMsg}")

                if headerCd == "7":
                    print("[CRITICAL] 호출횟수 초과 1시간 대기 진행중 프로그램을 1시간 내 중지 요망")
                    clear_all()

                    skip_record = True
                    time.sleep(3600)
                    continue

            location_items = location_root.findall(".//itemList")

            for item in location_items:
                vehId = item.findtext("vehId") #버스 ID
                sectOrd = int(item.findtext("sectOrd") or 0) #구간 순번
                sectDist = float(item.findtext("sectDist") or 0) #구간 옵셋 거리(구간 진행 거리)
                dataTm = item.findtext("dataTm") #제공시간
                plainNo = item.findtext("plainNo") #차량번호
                stopFlag = item.findtext("stopFlag") or "0" #정차여부
                fmt = "%Y%m%d%H%M%S"
                t_now = datetime.strptime(dataTm, fmt)

                #api 확장
                isrunyn = item.findtext("isrunyn")

                #호출시간 동기화 로직
                if last_processed_tm.get(vehId) == dataTm:
                    continue
                last_processed_tm[vehId] = dataTm

                #운행여부부분
                if isrunyn == "0":
                    if vehId in bus_state:
                        print(f"[운행종료] {vehId} - 상태 정리")
                        #진행중이던 세션 종료 처리
                        finalize_vehicle(vehId, sectOrd, dataTm, 0, jsonl_file)
                        clear_vehicle(vehId)
                    continue

                if vehId not in bus_state: #버스가 처음 등장 시
                    station = station_info.get(sectOrd)

                    bus_state[vehId] = {
                        "sectOrd": sectOrd,
                        "sectDist": sectDist,
                        "time": dataTm,
                        "stopFlag": stopFlag,
                        "trip_id": 0,
                        "trip_direction": station["direction"] if station else "UNKNOWN",

                        "isrunyn": isrunyn
                    }
                    continue

                prev = bus_state[vehId]
                travel_time_sec = 0
                
                curr_station = station_info.get(sectOrd)
                prev_station = station_info.get(prev["sectOrd"])

                if sectOrd - prev["sectOrd"] > 1:
                    print(f"[WARN] seq 점프 감지 {vehId} {prev['sectOrd']} -> {sectOrd}")
                    jump_count = jump_count + 1
                    finalize_vehicle(vehId, prev["sectOrd"] + 1, dataTm, 1, jsonl_file) #점프되었으니 prev["sectOrd"] 가 맞다

                    bus_state[vehId] = {
                        "sectOrd": sectOrd,
                        "sectDist": sectDist,
                        "time": dataTm,
                        "stopFlag": stopFlag,
                        "trip_id": prev["trip_id"],
                        "trip_direction": prev["trip_direction"],

                        "isrunyn": prev["isrunyn"]
                    }

                    session_dicts[vehId] = {
                        "vehId": vehId,
                        "fromSect": sectOrd,
                        "toSect": None,
                        "starttime": dataTm,
                        "endtime":None,
                        "10secDist": [],
                        "10secTm": []
                    }

                    continue
                
                if prev["sectOrd"] == meta["seq_max"] and sectOrd == meta["seq_min"]: #마지막 종점에서 시작 탐지
                    bus_state[vehId]["trip_id"] += 1
                    bus_state[vehId]["trip_direction"] = curr_station["direction"]

                if prev_station and curr_station: #회차발생여부탐지
                    if prev_station["direction"] != curr_station["direction"]:
                        bus_state[vehId]["trip_id"] += 1
                        bus_state[vehId]["trip_direction"] = curr_station["direction"] if curr_station else prev["trip_direction"]
                        last_arrival_time[vehId] = t_now

            
                if stopFlag == "1" and prev["stopFlag"] == "0" and not skip_record: #정류장 이동 감지 시
                    
                    session_dict = session_dicts.get(vehId)#기존 세션 종료
                    if session_dict and not skip_record:
                        session_dict["endtime"] = dataTm
                        session_dict["toSect"] = sectOrd
                        
                        #누적 기록 방식으로 기록
                        route_id = 100100033 #임의의 서울 160번
                        seq = session_dict['fromSect']
                        
                        record = {
                            "routeId": route_id,
                            "seq": seq,
                            "vehId": vehId,
                            "toSect": sectOrd,
                            "starttime": session_dict["starttime"],
                            "endtime": dataTm,
                            "is_jump": 0,
                            "10secDist": session_dict["10secDist"],
                            "10secTm": session_dict["10secTm"]
                        }

                    
                        jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                        jsonl_file.flush()

                        print(f"{vehId} 파일 기록 완료")
   
                    session_dicts[vehId] = {
                        "vehId": vehId,
                        "fromSect": sectOrd,
                        "toSect": None,
                        "starttime": dataTm,
                        "endtime":None,
                        "10secDist": [],
                        "10secTm": []
                    }


                    station = station_info.get(sectOrd)
                    
                    
                    
                    if vehId in last_arrival_time:
                        travel_time_sec = int((t_now - last_arrival_time[vehId]).total_seconds())
                    else:
                        travel_time_sec = -1 #측정이 시작되는 경우, “-1은 ‘계산 불가능한 초기값’이라 분석에서 제외해야 한다”!!
                
                    print(f"{datetime.now()} - 정류장 이동: {vehId} {prev['sectOrd']} -> {sectOrd}")
                    
                    writer.writerow({
                        "vehId": vehId,
                        "plainNo": plainNo,

                        "fromSect": prev["sectOrd"],
                        "toSect": sectOrd,

                        "starttime": prev["time"], # 나중에 실제 출발  시점 따로 잡는 로직 필요 현재는 괜찮다
                        "endtime": dataTm,

                        "prevSectDist": prev["sectDist"],#10초 전 구간 옵셋 거리
                        "currSectDist": sectDist, #현재 구간 옵셋 거리(두개의 차로 정지 구분 가능)

                        "trip_id": bus_state[vehId]["trip_id"],
                        "trip_direction": bus_state[vehId]["trip_direction"],

                        "seq": sectOrd,
                        "stationName": station["name"] if station else "UNKNOWN",

                        "travelTimeSec": travel_time_sec
                    }) #방향 direction 추가할것, 회차 여부도 추가할것 설계 먼저 하고 추가

                    csvfile.flush() #바로 파일 기록

                if stopFlag == "0" and prev["stopFlag"] == "1" and vehId in session_dicts:
                    session_dicts[vehId]["starttime"] = dataTm
                    last_arrival_time[vehId] = t_now

                         
                    session_dicts[vehId]["10secDist"].append(sectDist) #starttime 수정 후 데이터 공란 에러 매우 많이 발생, 수정함
                    session_dicts[vehId]["10secTm"].append(dataTm) #starttime 수정 후 데이터 공란 에러 매우 많이 발생, 수정함

                    print(f"{vehId} 출발 감지, starttime 갱신: {dataTm}")

                elif stopFlag == "0" and prev["stopFlag"] == "0" and vehId in session_dicts:

                    if sectOrd != prev["sectOrd"]: #구간 바뀌면 무시
                        continue

                    prev_dist = prev["sectDist"]

                    if sectDist < prev_dist:
                        continue

                    #연속 15회 이상 같은 값 체크
                    recent_distances = session_dicts[vehId]["10secDist"][-14:] #마지막 14개 가져오기
                    if len(recent_distances) >= 14 and all(d == sectDist for d in recent_distances):
                        print(f"{vehId} - 동일 거리 {sectDist} 15회 이상, 가록하지 않음")
                        continue

                    session_dicts[vehId]["10secDist"].append(sectDist)
                    session_dicts[vehId]["10secTm"].append(dataTm)
                    print(f"{vehId} 이동거리 추가 , 누적: {session_dicts[vehId]['10secDist']}")



                #상태 업데이트
                bus_state[vehId] = {
                    "sectOrd": sectOrd,
                    "sectDist": sectDist,
                    "time": dataTm,
                    "stopFlag": stopFlag,
                    "trip_id": prev["trip_id"],
                    "trip_direction": prev["trip_direction"],
                    "isrunyn": isrunyn
                }

            time.sleep(4) #공식문서에 5초라고 되어있음

    except KeyboardInterrupt:
        csvfile.flush() #바로 파일 기록
        jsonl_file.close()
        print(f"{jump_count} : 점프 횟수")
        print("\n수집 중지, 파일 저장 완료")