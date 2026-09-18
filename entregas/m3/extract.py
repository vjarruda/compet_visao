"""
Extração de keypoints (pontos-chave) de mãos e corpo a partir de um vídeo.

Primeiro passo do pipeline do PneumoVision (RF-03 no PRD): recebe um vídeo de
um paciente usando o inalador e produz, para cada quadro, os pontos-chave das
mãos (Hand Landmarker) e do corpo (Pose Landmarker) via MediaPipe Tasks API.

Trabalhar sobre keypoints — e não sobre a imagem bruta — reduz a necessidade de
dados, melhora a generalização e preserva a privacidade do paciente.

Uso:
    python -m pipeline.keypoints.extract --input data/samples/exemplo.mp4 \
        --output data/keypoints/exemplo.json

Na primeira execução, os modelos .task (hand_landmarker, pose_landmarker) são
baixados automaticamente para ./models/ (ou para --models-dir).

Saída (JSON):
    {
      "metadata": { ... informações do vídeo e da extração ... },
      "frames": [ {"frame": 0, "timestamp_ms": 0, "hands": [...], "pose": [...]}, ... ]
    }

Cada landmark segue o schema do PRD: {"landmark": i, "x", "y", "z", "score"}.
Para as mãos, "score" é a confiança de classificação da mão (handedness);
para o corpo, "score" é a visibilidade do ponto (visibility).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
import uuid
from dataclasses import dataclass

try:
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    from tqdm import tqdm
except ImportError as e:  # mensagem amigável para quem está começando
    print(
        "Dependência ausente:", e,
        "\nInstale as dependências do projeto com: pip install -r requirements.txt",
        file=sys.stderr,
    )
    raise

# URLs oficiais dos modelos MediaPipe (bundles .task).
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)


@dataclass
class ExtractionConfig:
    input_path: str
    output_path: str
    models_dir: str = "models"
    max_hands: int = 2
    detect_pose: bool = True
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


# --------------------------------------------------------------------------- #
# Modelos
# --------------------------------------------------------------------------- #
def ensure_model(url: str, models_dir: str) -> str:
    """Garante que o modelo .task exista localmente; baixa se necessário."""
    os.makedirs(models_dir, exist_ok=True)
    dest = os.path.join(models_dir, os.path.basename(url))
    if not os.path.isfile(dest):
        print(f"Baixando modelo: {os.path.basename(dest)} ...", file=sys.stderr)
        urllib.request.urlretrieve(url, dest)
    return dest


# --------------------------------------------------------------------------- #
# Conversão de landmarks para o schema do PRD
# --------------------------------------------------------------------------- #
def _hand_to_dict(hand_landmarks, hand_score: float) -> list[dict]:
    return [
        {
            "landmark": i,
            "x": round(lm.x, 6),
            "y": round(lm.y, 6),
            "z": round(lm.z, 6),
            "score": round(float(hand_score), 4),
        }
        for i, lm in enumerate(hand_landmarks)
    ]


def _pose_to_dict(pose_landmarks) -> list[dict]:
    return [
        {
            "landmark": i,
            "x": round(lm.x, 6),
            "y": round(lm.y, 6),
            "z": round(lm.z, 6),
            "score": round(float(getattr(lm, "visibility", 0.0)), 4),
        }
        for i, lm in enumerate(pose_landmarks)
    ]


def _build_hand_landmarker(model_path: str, cfg: ExtractionConfig):
    opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=cfg.max_hands,
        min_hand_detection_confidence=cfg.min_detection_confidence,
        min_tracking_confidence=cfg.min_tracking_confidence,
    )
    return mp_vision.HandLandmarker.create_from_options(opts)


def _build_pose_landmarker(model_path: str, cfg: ExtractionConfig):
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=mp_vision.RunningMode.VIDEO,
        min_pose_detection_confidence=cfg.min_detection_confidence,
        min_tracking_confidence=cfg.min_tracking_confidence,
    )
    return mp_vision.PoseLandmarker.create_from_options(opts)


# --------------------------------------------------------------------------- #
# Extração
# --------------------------------------------------------------------------- #
def extract_keypoints(config: ExtractionConfig) -> dict:
    if not os.path.isfile(config.input_path):
        raise FileNotFoundError(f"Vídeo não encontrado: {config.input_path}")

    hand_model = ensure_model(HAND_MODEL_URL, config.models_dir)
    pose_model = ensure_model(POSE_MODEL_URL, config.models_dir) if config.detect_pose else None

    cap = cv2.VideoCapture(config.input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {config.input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    hand_landmarker = _build_hand_landmarker(hand_model, config)
    pose_landmarker = _build_pose_landmarker(pose_model, config) if pose_model else None

    frames: list[dict] = []
    frames_with_hands = 0
    frames_with_pose = 0
    frame_idx = 0

    try:
        with tqdm(total=total_frames or None, desc="Extraindo keypoints", unit="quadro") as bar:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                # timestamp monotônico em ms (exigido pelo modo VIDEO)
                ts_ms = int(frame_idx * (1000.0 / fps)) if fps > 0 else frame_idx

                # ---- Mãos ----
                hands_data: list[dict] = []
                hand_result = hand_landmarker.detect_for_video(mp_image, ts_ms)
                if hand_result.hand_landmarks:
                    frames_with_hands += 1
                    for h_idx, hand_landmarks in enumerate(hand_result.hand_landmarks):
                        label, score = "Unknown", 0.0
                        if hand_result.handedness and h_idx < len(hand_result.handedness):
                            cls = hand_result.handedness[h_idx][0]
                            label, score = cls.category_name, cls.score
                        hands_data.append(
                            {"hand": label, "landmarks": _hand_to_dict(hand_landmarks, score)}
                        )

                # ---- Corpo ----
                pose_data: list[dict] = []
                if pose_landmarker is not None:
                    pose_result = pose_landmarker.detect_for_video(mp_image, ts_ms)
                    if pose_result.pose_landmarks:
                        frames_with_pose += 1
                        pose_data = _pose_to_dict(pose_result.pose_landmarks[0])

                frames.append(
                    {"frame": frame_idx, "timestamp_ms": ts_ms, "hands": hands_data, "pose": pose_data}
                )
                frame_idx += 1
                bar.update(1)
    finally:
        cap.release()
        hand_landmarker.close()
        if pose_landmarker is not None:
            pose_landmarker.close()

    processed = frame_idx or 1
    return {
        "metadata": {
            "video_id": str(uuid.uuid4()),
            "source_file": os.path.basename(config.input_path),
            "fps": round(fps, 2),
            "frames_declared": total_frames,
            "frames_processed": frame_idx,
            "hand_detection_rate": round(frames_with_hands / processed, 4),
            "pose_detection_rate": round(frames_with_pose / processed, 4),
            "config": {
                "max_hands": config.max_hands,
                "detect_pose": config.detect_pose,
                "min_detection_confidence": config.min_detection_confidence,
                "min_tracking_confidence": config.min_tracking_confidence,
            },
        },
        "frames": frames,
    }


def save_json(data: dict, output_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extrai keypoints de mãos e corpo de um vídeo (MediaPipe Tasks API)."
    )
    parser.add_argument("--input", required=True, help="Caminho do vídeo de entrada.")
    parser.add_argument("--output", required=True, help="Caminho do JSON de saída.")
    parser.add_argument("--models-dir", default="models", help="Pasta dos modelos .task.")
    parser.add_argument("--max-hands", type=int, default=2, help="Máximo de mãos a detectar.")
    parser.add_argument("--no-pose", action="store_true", help="Desativa a extração de pose.")
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = ExtractionConfig(
        input_path=args.input,
        output_path=args.output,
        models_dir=args.models_dir,
        max_hands=args.max_hands,
        detect_pose=not args.no_pose,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    result = extract_keypoints(config)
    save_json(result, config.output_path)

    meta = result["metadata"]
    print(
        f"\n\u2713 Extração concluída: {meta['frames_processed']} quadros"
        f"\n  Arquivo: {config.output_path}"
        f"\n  Detecção de mãos:  {meta['hand_detection_rate'] * 100:.1f}% dos quadros"
        f"\n  Detecção de pose:  {meta['pose_detection_rate'] * 100:.1f}% dos quadros"
    )
    if meta["hand_detection_rate"] < 0.90:
        print(
            "  \u26a0\ufe0f  Detecção de mãos abaixo da meta de 90% (M2 do PRD) — verifique "
            "iluminação, enquadramento e qualidade do vídeo (ver protocolo de captura)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
