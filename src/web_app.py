"""
Веб-приложение для LighthouseForCycles на FastAPI.

Функции:
- потоковое MJPEG-видео с камеры;
- API конфигурации (номер камеры, лимит скорости);
- API состояния «лампочки» (превышена ли скорость).
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.camera import Camera, CameraError, load_config
from src.speed_utils import compute_speed_kmh


class _AppState:
    """Глобальное состояние для веб-приложения."""

    def __init__(self) -> None:
        self.frame_lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None

        self.lamp_lock = threading.Lock()
        self.is_speed_exceeded: bool = False

        self.config_lock = threading.Lock()
        self.config: Dict[str, Any] = {}
        self.config_path: Optional[Path] = None

        self.worker_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        self.current_camera_index: Optional[int] = None


state = _AppState()


def _load_initial_config() -> None:
    """Загрузить конфигурацию с диска в память."""
    config = load_config()
    project_root = Path(__file__).parent.parent
    config_path = project_root / "config.json"

    with state.config_lock:
        state.config = config
        state.config_path = config_path
        camera_cfg = config.get("camera", {})
        state.current_camera_index = int(camera_cfg.get("index", 0))


def _save_config_to_disk(config: Dict[str, Any]) -> None:
    """Сохранить конфигурацию в config.json."""
    config_path = state.config_path
    if config_path is None:
        project_root = Path(__file__).parent.parent
        config_path = project_root / "config.json"

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def _update_frame(frame: np.ndarray) -> None:
    with state.frame_lock:
        state.latest_frame = frame.copy()


def _update_lamp(is_exceeded: bool) -> None:
    with state.lamp_lock:
        state.is_speed_exceeded = is_exceeded


def _get_current_config() -> Dict[str, Any]:
    with state.config_lock:
        # Глубокое копирование без ссылок на исходный словарь.
        copied: Dict[str, Any] = json.loads(json.dumps(state.config))
        return copied


def _camera_worker() -> None:
    """
    Фоновый поток: читает кадры с камеры, выполняет детекцию и обновляет состояние.

    Для простоты дублирует часть логики display_video_stream, но без окон OpenCV.
    """
    try:
        while not state.stop_event.is_set():
            config = _get_current_config()
            camera_cfg = config.get("camera", {})
            detection_cfg = config.get("detection", {})
            speed_limit_kmh = float(config.get("speed_limit_kmh", 8.0))
            # Шаг по кадрам для детекции: 1 — детектить каждый кадр,
            # 2 — каждый второй, 3 — каждый третий и т.д.
            # Значение <= 0 принудительно заменяем на 1.
            detection_frame_stride = int(config.get("detection_frame_stride", 1) or 1)
            if detection_frame_stride <= 0:
                detection_frame_stride = 1
            red_hold_seconds = float(config.get("red_hold_seconds", 2.0))

            camera_index = int(camera_cfg.get("index", 0))
            width = camera_cfg.get("width")
            height = camera_cfg.get("height")
            fps_val = camera_cfg.get("fps")

            detection_enabled = bool(detection_cfg.get("enabled", False))

            # Инициализируем YOLO-детектор, если включён.
            yolo_detector = None
            if detection_enabled:
                try:
                    from src.detector import DetectionConfig, YOLOPeopleDetector

                    det_cfg = DetectionConfig(
                        model_path=detection_cfg.get("model_path", "yolo11n.pt"),
                        conf=float(detection_cfg.get("conf", 0.5)),
                        device=detection_cfg.get("device"),
                        imgsz=int(detection_cfg.get("imgsz", 640)),
                    )
                    yolo_detector = YOLOPeopleDetector(det_cfg)
                    if not yolo_detector.is_available():
                        detection_enabled = False
                except Exception:  # noqa: BLE001
                    detection_enabled = False

            camera = Camera(index=camera_index, width=width, height=height, fps=fps_val)
            try:
                camera.connect()
            except CameraError:
                # Если камера не доступна — ждём и пробуем ещё раз.
                _update_lamp(False)
                time.sleep(1.0)
                continue

            # Хранилище треков:
            # track_id -> {
            #   "last_pos": (cx, cy),
            #   "last_time": float,
            #   "speed_kmh": Optional[float],
            #   "is_over_limit": bool,
            #   "last_over_limit_time": Optional[float],
            # }
            tracks: Dict[int, Dict[str, Any]] = {}
            track_ttl_seconds = 2.0
            frame_idx = 0

            # Последнее состояние «превышения» для лампы, чтобы не дёргать
            # детекцию на каждом кадре, но сохранять визуальное состояние.
            last_is_exceeded_any = False

            try:
                while not state.stop_event.is_set():
                    ret, frame = camera.read_frame()
                    if not ret or frame is None:
                        break

                    frame_idx += 1
                    now = time.time()

                    # По умолчанию берём предыдущее состояние лампы.
                    is_exceeded_any = last_is_exceeded_any

                    # Детекцию выполняем не на каждом кадре, а с заданным шагом,
                    # чтобы уменьшить нагрузку на CPU/GPU.
                    if (
                        detection_enabled
                        and yolo_detector is not None
                        and frame_idx % detection_frame_stride == 0
                    ):
                        try:
                            detections = yolo_detector.detect(frame)
                        except Exception:  # noqa: BLE001
                            detections = []
                            detection_enabled = False

                        # Очистка устаревших треков
                        if frame_idx % 30 == 0:
                            expired_ids = []
                            for track_id, data in tracks.items():
                                last_time_val = data.get("last_time")
                                if isinstance(last_time_val, (int, float)) and now - last_time_val > track_ttl_seconds:
                                    expired_ids.append(track_id)
                            for track_id in expired_ids:
                                tracks.pop(track_id, None)

                        for det in detections:
                            det_track_id: Optional[int]
                            if len(det) == 5:
                                x1, y1, x2, y2, score = det
                                det_track_id = None
                            elif len(det) == 6:
                                x1, y1, x2, y2, score, det_track_id = det
                            else:
                                continue

                            cx = (x1 + x2) / 2.0
                            cy = (y1 + y2) / 2.0

                            speed_kmh: Optional[float] = None
                            prev: Dict[str, Any] = {}

                            if det_track_id is not None and det_track_id >= 0:
                                prev = tracks.get(det_track_id, {})
                                last_pos = prev.get("last_pos")
                                last_time_val = prev.get("last_time")
                                if (
                                    isinstance(last_pos, tuple)
                                    and len(last_pos) == 2
                                    and isinstance(last_time_val, (int, float))
                                ):
                                    speed_kmh = compute_speed_kmh(
                                        last_pos,
                                        float(last_time_val),
                                        (cx, cy),
                                        now,
                                    )

                                prev_speed = prev.get("speed_kmh")
                                current_speed = speed_kmh if speed_kmh is not None else prev_speed

                                is_over_limit_prev = bool(prev.get("is_over_limit", False))
                                last_over_limit_time_prev = prev.get("last_over_limit_time")

                                if current_speed is not None and current_speed > speed_limit_kmh:
                                    is_over_limit = True
                                    last_over_limit_time = now
                                else:
                                    if (
                                        is_over_limit_prev
                                        and isinstance(last_over_limit_time_prev, (int, float))
                                        and now - float(last_over_limit_time_prev) <= red_hold_seconds
                                    ):
                                        is_over_limit = True
                                        last_over_limit_time = last_over_limit_time_prev
                                    else:
                                        is_over_limit = False
                                        last_over_limit_time = last_over_limit_time_prev

                                tracks[det_track_id] = {
                                    "last_pos": (cx, cy),
                                    "last_time": now,
                                    "speed_kmh": current_speed,
                                    "is_over_limit": is_over_limit,
                                    "last_over_limit_time": last_over_limit_time,
                                }
                            else:
                                det_track_id = None
                                current_speed = None
                                is_over_limit = False

                            if det_track_id is not None:
                                track_data = tracks.get(det_track_id, {})
                                label_speed = track_data.get("speed_kmh")
                                is_over_limit = bool(track_data.get("is_over_limit", False))
                            else:
                                label_speed = None

                            # Если есть хотя бы один трек с превышением (с учётом задержки) —
                            # считаем, что маяк должен быть красным.
                            if is_over_limit:
                                is_exceeded_any = True

                            # Цвет рамки и текста
                            box_color = (0, 255, 0)
                            text_color = (0, 255, 0)
                            if is_over_limit:
                                box_color = (0, 0, 255)
                                text_color = (0, 0, 255)

                            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                            if label_speed is not None:
                                label = f"person {score:.2f} | {label_speed:.1f} km/h"
                            else:
                                label = f"person {score:.2f}"

                            cv2.putText(
                                frame,
                                label,
                                (x1, max(y1 - 10, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                text_color,
                                2,
                            )

                        # Обновляем запомненное состояние лампы только после детекции.
                        last_is_exceeded_any = is_exceeded_any

                    _update_lamp(is_exceeded_any)
                    _update_frame(frame)

                    time.sleep(0.001)

                    # Проверяем, не поменялся ли номер камеры в конфиге.
                    with state.config_lock:
                        new_index = int(state.config.get("camera", {}).get("index", camera_index))
                    if new_index != camera_index:
                        break

            finally:
                camera.release()

            # Небольшая пауза перед возможной следующей попыткой.
            time.sleep(0.1)

    finally:
        _update_lamp(False)


def _start_worker_if_needed() -> None:
    if state.worker_thread is not None and state.worker_thread.is_alive():
        return

    state.stop_event.clear()
    worker = threading.Thread(target=_camera_worker, name="camera-worker", daemon=True)
    state.worker_thread = worker
    worker.start()


def create_app() -> FastAPI:
    """Создать и настроить экземпляр FastAPI-приложения."""
    app = FastAPI(title="LighthouseForCycles Web")

    _load_initial_config()
    _start_worker_if_needed()

    dist_dir = Path(__file__).parent.parent / "frontend" / "dist"
    if dist_dir.exists():
        app.mount(
            "/assets",
            StaticFiles(directory=dist_dir / "assets"),
            name="assets",
        )

        @app.get("/", response_class=HTMLResponse)
        async def index() -> HTMLResponse:
            index_path = dist_dir / "index.html"
            if index_path.exists():
                return HTMLResponse(index_path.read_text(encoding="utf-8"))
            return HTMLResponse("<html><body><h1>LighthouseForCycles</h1></body></html>")
    else:
        @app.get("/", response_class=HTMLResponse)
        async def index_placeholder() -> HTMLResponse:
            return HTMLResponse(
                "<html><body>"
                "<h1>LighthouseForCycles</h1>"
                "<p>React-билд пока не собран. Соберите фронтенд и поместите его в frontend/dist.</p>"
                "</body></html>"
            )

    @app.get("/api/config")
    async def get_config() -> Dict[str, Any]:
        return _get_current_config()

    @app.patch("/api/config")
    async def patch_config(payload: Dict[str, Any]) -> Dict[str, Any]:
        with state.config_lock:
            config = state.config

            camera_cfg = config.setdefault("camera", {})
            if "camera" in payload:
                incoming_camera = payload["camera"]
                if isinstance(incoming_camera, dict):
                    for key in ("index", "width", "height", "fps"):
                        if key in incoming_camera:
                            camera_cfg[key] = incoming_camera[key]

            if "speed_limit_kmh" in payload:
                try:
                    config["speed_limit_kmh"] = float(payload["speed_limit_kmh"])
                except (TypeError, ValueError) as exc:
                    raise HTTPException(status_code=400, detail="speed_limit_kmh must be a number") from exc

            if "red_hold_seconds" in payload:
                try:
                    config["red_hold_seconds"] = float(payload["red_hold_seconds"])
                except (TypeError, ValueError) as exc:
                    raise HTTPException(status_code=400, detail="red_hold_seconds must be a number") from exc

            state.config = config
            _save_config_to_disk(config)

        return _get_current_config()

    @app.get("/api/lamp")
    async def get_lamp_state() -> Dict[str, str]:
        with state.lamp_lock:
            lamp = "red" if state.is_speed_exceeded else "green"
        return {"lamp": lamp}

    def frame_generator() -> Iterator[bytes]:
        boundary = b"--frame"
        while True:
            if state.stop_event.is_set():
                break

            with state.frame_lock:
                frame = state.latest_frame.copy() if state.latest_frame is not None else None

            if frame is None:
                time.sleep(0.05)
                continue

            ok, encoded = cv2.imencode(".jpg", frame)
            if not ok:
                time.sleep(0.01)
                continue

            jpg_bytes = encoded.tobytes()
            yield (
                boundary
                + b"\r\nContent-Type: image/jpeg\r\n\r\n"
                + jpg_bytes
                + b"\r\n"
            )

    @app.get("/stream")
    async def stream() -> StreamingResponse:
        return StreamingResponse(
            frame_generator(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.on_event("shutdown")
    async def on_shutdown() -> None:
        state.stop_event.set()
        worker = state.worker_thread
        if worker is not None:
            worker.join(timeout=5.0)

    return app


app = create_app()
