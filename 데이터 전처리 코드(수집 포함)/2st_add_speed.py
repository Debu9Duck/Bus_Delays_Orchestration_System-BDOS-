import json
from datetime import datetime

input_path  = "3st_input.jsonl"
output_path = "3st_output.jsonl"

def has_excessive_repeats(dist_list, threshold=50):
    if not dist_list:
        return False
    
    current_count = 1
    for i in range(1, len(dist_list)):
        if dist_list[i] == dist_list[i-1]:
            current_count += 1
            if current_count >= threshold:
                return True
        else:
            current_count = 1
    return False

def calc_speed(dist_list, tm_list):
    FMT = "%Y%m%d%H%M%S"
    speeds = []

    for i in range(len(dist_list)):
        if i == 0:
            speeds.append(0.0)
        else:
            dd = dist_list[i] - dist_list[i - 1]
            
            try:
                t0 = datetime.strptime(tm_list[i - 1], FMT)
                t1 = datetime.strptime(tm_list[i], FMT)
                dt = (t1 - t0).total_seconds()
            except (ValueError, IndexError):
                dt = 0

            if dt <= 0:
                speeds.append(0.0)
            else:
                speed = dd / dt * 3600
                speeds.append(round(speed, 2))
    return speeds

def process_bus_data(input_path, output_path):
    total_count = 0
    saved_count = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            total_count += 1
            
            dist_list = record.get("10secDist", [])
            tm_list   = record.get("10secTm",   [])

            # 동일 값 35회 이상 반복 체크 (연속성 기준)
            if has_excessive_repeats(dist_list, threshold=35):
                continue  # 조건에 걸리면 저장하지 않고 다음 줄로 넘어감

            # 속도 계산 및 추가
            speeds = calc_speed(dist_list, tm_list)
            record["10secSpeed"] = speeds

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            saved_count += 1

    print(f"전체 레코드: {total_count}개")
    print(f"저장된 레코드 (반복 제거 후): {saved_count}개")
    print(f"삭제된 레코드: {total_count - saved_count}개")

if __name__ == "__main__":
    process_bus_data(input_path, output_path)