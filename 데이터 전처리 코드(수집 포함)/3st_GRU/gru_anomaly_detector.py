import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import joblib

# ──────────────────────────────────────────
# 0. JSONL 데이터 로드
# ──────────────────────────────────────────
JSONL_PATH = Path(__file__).parent / "input_GRU.jsonl"

def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        print(f"[ERROR] 데이터 파일을 찾을 수 없습니다: {path}")
        print("  → input_GRU.jsonl 파일을 스크립트와 같은 폴더에 두고 다시 실행하세요.")
        sys.exit(1)

    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] {line_no}번째 줄 파싱 오류 (건너뜀): {e}")

    if not records:
        print("[ERROR] JSONL 파일에 유효한 레코드가 없습니다.")
        sys.exit(1)

    return records


def validate_record(r: dict, idx: int) -> bool:
    required = ["vehId", "seq", "10secSpeed", "10secDist"]
    for key in required:
        if key not in r:
            print(f"[WARN] 레코드 #{idx} — '{key}' 필드 누락, 건너뜁니다.")
            return False
    if not r["10secSpeed"] or not r["10secDist"]:
        print(f"[WARN] 레코드 #{idx} (vehId={r.get('vehId')}) — 속도/거리 배열이 비어있습니다.")
        return False
    return True


RAW_DATA = load_jsonl(JSONL_PATH)
RAW_DATA = [r for i, r in enumerate(RAW_DATA) if validate_record(r, i)]

# ──────────────────────────────────────────
# 1. 피처 엔지니어링
# ──────────────────────────────────────────
def extract_features(record: dict) -> np.ndarray:
    speeds = np.array(record["10secSpeed"], dtype=np.float32)
    dists  = np.array(record["10secDist"],  dtype=np.float32)
    speed_diff = np.diff(speeds, prepend=speeds[0])
    dist_diff  = np.diff(dists,  prepend=dists[0])
    return np.stack([speeds, speed_diff, dist_diff], axis=1)


def sliding_windows(seq: np.ndarray, window: int = 8, stride: int = 1) -> list:
    T = len(seq)
    if T <= window:
        pad = np.zeros((window - T, seq.shape[1]), dtype=np.float32)
        return [np.concatenate([seq, pad], axis=0)]
    shape   = ((T - window) // stride + 1, window, seq.shape[1])
    strides = (seq.strides[0] * stride, seq.strides[0], seq.strides[1])
    windows = np.lib.stride_tricks.as_strided(seq, shape=shape, strides=strides)
    return list(windows)


# ──────────────────────────────────────────
# 2. PyTorch Dataset
# ──────────────────────────────────────────
class TrajectoryDataset(Dataset):
    def __init__(self, sequences: list):
        arr = np.stack(sequences, axis=0)
        self.data = torch.from_numpy(arr).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ──────────────────────────────────────────
# 3. GRU Autoencoder 모델
# ──────────────────────────────────────────
class GRUEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc  = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        _, h_n = self.gru(x)
        return self.fc(h_n[-1])


class GRUDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int, seq_len: int):
        super().__init__()
        self.seq_len = seq_len
        self.fc  = nn.Linear(latent_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = self.fc(z).unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.gru(h)
        return self.out(out)


class GRUAutoencoder(nn.Module):
    def __init__(self, input_dim: int = 3, hidden_dim: int = 32,
                 latent_dim: int = 16, seq_len: int = 8):
        super().__init__()
        self.encoder = GRUEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = GRUDecoder(latent_dim, hidden_dim, input_dim, seq_len)

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ──────────────────────────────────────────
# 4. 이상 점수 계산 (step별 평균 MSE)
# ──────────────────────────────────────────
def calculate_scores(model: GRUAutoencoder, seq_tensor: torch.Tensor,
                     device: torch.device = torch.device("cpu")) -> list[float]:
    model.eval()
    WINDOW = model.decoder.seq_len
    T = seq_tensor.shape[0]

    with torch.no_grad():
        if T <= WINDOW:
            pad = torch.zeros(WINDOW - T, seq_tensor.shape[1])
            x   = torch.cat([seq_tensor, pad], dim=0).unsqueeze(0).to(device)
            err = ((x - model(x)) ** 2).mean(dim=-1).squeeze()
            return err[:T].cpu().tolist()

        count  = np.zeros(T, dtype=np.float32)
        accum  = np.zeros(T, dtype=np.float32)
        for start in range(0, T - WINDOW + 1):
            x   = seq_tensor[start:start + WINDOW].unsqueeze(0).to(device)
            err = ((x - model(x)) ** 2).mean(dim=-1).squeeze().cpu().numpy()
            accum[start:start + WINDOW] += err
            count[start:start + WINDOW] += 1

        safe_count = np.where(count == 0, 1, count)
        return (accum / safe_count).tolist()


# ──────────────────────────────────────────
# 5 & 6. 학습 + 결과 빌드
# ──────────────────────────────────────────
WINDOW      = 8
FEATURES    = 3
EPOCHS      = 150
LR          = 1e-3
BATCH       = 512
NUM_WORKERS = 4

MODEL_PATH   = Path(__file__).parent / "gru_model.pt"
SCALER_PATH  = Path(__file__).parent / "gru_scaler.pkl"
THRESH_PATH  = Path(__file__).parent / "gru_threshold.pkl"

ANOMALY_SCORE_CUTOFF = 0.0001
OUTPUT_JSONL_PATH    = Path(__file__).parent / "output_GRU.jsonl"

processed_data: list = []
THRESHOLD: float     = 0.0


def _get_device() -> torch.device:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f">> GPU 사용: {torch.cuda.get_device_name(0)}  "
              f"(VRAM {torch.cuda.get_device_properties(0).total_memory // 1024**2} MB)", flush=True)
    else:
        print(">> CUDA를 찾을 수 없습니다. CPU로 실행합니다.", flush=True)
    return DEVICE


def _build_scaled(raw_data) -> tuple:
    print(">> 피처 추출 및 정규화 중...", flush=True)
    all_features = [extract_features(r) for r in raw_data]
    scaler = MinMaxScaler()
    scaler.fit(np.concatenate(all_features, axis=0))
    scaled = [scaler.transform(f) for f in all_features]
    return scaler, scaled


def _run_inference(model, scaled, raw_data, device) -> tuple:
    print(">> 이상 점수 계산 중...", flush=True)
    results = []
    for record, sc in zip(raw_data, scaled):
        scores    = calculate_scores(model, torch.tensor(sc, dtype=torch.float32), device)
        speeds    = record["10secSpeed"]
        max_delta = float(np.max(np.abs(np.diff(speeds, prepend=speeds[0]))))
        results.append({
            "vehId":     record["vehId"],
            "seq":       record["seq"],
            "speeds":    speeds,
            "maxSpd":    float(np.max(speeds)),
            "maxDel":    max_delta,
            "scores":    scores,
            "max_score": float(np.max(scores)),
        })
    all_scores_flat = np.concatenate([p["scores"] for p in results])
    threshold       = float(np.percentile(all_scores_flat, 95))
    print(f">> 이상 임계값 (95th percentile): {threshold:.6f}", flush=True)
    return results, threshold


def export_filtered_jsonl(raw_data: list, results: list) -> None:
    score_map = {(r["vehId"], r["seq"]): r["max_score"] for r in results}

    normal_records = []
    anomaly_count  = 0
    for record in raw_data:
        key   = (record["vehId"], record["seq"])
        score = score_map.get(key, 0.0)
        if score >= ANOMALY_SCORE_CUTOFF:
            anomaly_count += 1
        else:
            normal_records.append(record)

    with OUTPUT_JSONL_PATH.open("w", encoding="utf-8") as f:
        for record in normal_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    total = len(raw_data)
    print(f"완료: 전체 {total}개 | 정상 {len(normal_records)}개 | 기각 {anomaly_count}개 "
          f"(기준: max_score ≥ {ANOMALY_SCORE_CUTOFF})", flush=True)


def train_and_save() -> None:
    global processed_data, THRESHOLD
    DEVICE = _get_device()

    scaler, scaled = _build_scaled(RAW_DATA)

    print(f">> 슬라이딩 윈도우 생성 중 (레코드 {len(RAW_DATA):,}개) ...", flush=True)
    all_windows = []
    for s in scaled:
        all_windows.extend(sliding_windows(s, window=WINDOW))

    dataset = TrajectoryDataset(all_windows)
    del all_windows

    loader = DataLoader(
        dataset,
        batch_size=BATCH,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=True,
    )

    torch.manual_seed(42)
    model     = GRUAutoencoder(input_dim=FEATURES, hidden_dim=32, latent_dim=16, seq_len=WINDOW).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    print(f">> GRU Autoencoder 학습 시작 (epochs={EPOCHS}, lr={LR}, device={DEVICE}) ...", flush=True)
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for batch in loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(batch), batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg = epoch_loss / len(loader)
        print(f"   Epoch [{epoch+1:3d}/{EPOCHS}] — avg loss: {avg:.6f}", flush=True)

    print(">> 학습 완료.", flush=True)

    torch.save(model.state_dict(), MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f">> 모델 저장 완료: {MODEL_PATH}", flush=True)

    results, threshold = _run_inference(model, scaled, RAW_DATA, DEVICE)
    joblib.dump(threshold, THRESH_PATH)
    print(f">> 임계값 저장 완료: {THRESH_PATH}", flush=True)

    processed_data = results
    THRESHOLD      = threshold

    export_filtered_jsonl(RAW_DATA, results)


def load_and_infer() -> None:
    global processed_data, THRESHOLD
    DEVICE = _get_device()

    print(f">> 저장된 모델 로드: {MODEL_PATH}", flush=True)
    model = GRUAutoencoder(input_dim=FEATURES, hidden_dim=32, latent_dim=16, seq_len=WINDOW)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)

    scaler    = joblib.load(SCALER_PATH)
    THRESHOLD = joblib.load(THRESH_PATH)
    print(f">> 스케일러·임계값 로드 완료. 임계값={THRESHOLD:.6f}", flush=True)

    print(">> 피처 추출 및 정규화 중...", flush=True)
    all_features = [extract_features(r) for r in RAW_DATA]
    scaled = [scaler.transform(f) for f in all_features]

    results, _ = _run_inference(model, scaled, RAW_DATA, DEVICE)
    processed_data = results

    export_filtered_jsonl(RAW_DATA, results)


def run_training() -> None:
    if MODEL_PATH.exists() and SCALER_PATH.exists() and THRESH_PATH.exists():
        print(">> 저장된 모델 파일 발견 → 학습 없이 추론 모드로 실행합니다.", flush=True)
        print("   (재학습하려면 gru_model.pt / gru_scaler.pkl / gru_threshold.pkl 을 삭제하세요.)", flush=True)
        load_and_infer()
    else:
        print(">> 저장된 모델 없음 → 학습 후 저장합니다.", flush=True)
        train_and_save()


if __name__ == "__main__":
    run_training()