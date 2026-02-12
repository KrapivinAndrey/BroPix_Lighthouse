"""
Веб-приложение для LighthouseForCycles на FastAPI.

Функции:
- потоковое MJPEG-видео с камеры;
- API конфигурации (номер камеры, лимит скорости);
- API состояния «лампочки» (превышена ли скорость).
"""

from __future__ import annotations

import json
import logging
import signal
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
from src.config.service import ConfigService
from src.lighthouse.implementations import UILighthouseController
from src.processing.frame_processor import FrameProcessor
from src.speed_utils import DEFAULT_PX_TO_M_SCALE, compute_speed_kmh


logger = logging.getLogger(__name__)


class AppState:
    """
    Состояние веб-приложения с явными зависимостями.

    Зависимости передаются через конструктор (dependency injection),
    что упрощает тестирование и масштабирование.
    """

    def __init__(
        self,
        config_service: ConfigService,
        lighthouse_controller: UILighthouseController,
    ) -> None:
        self.config_service = config_service
        self.lighthouse_controller = lighthouse_controller

        self.frame_lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None

        self.worker_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()


# Глобальное состояние для обратной совместимости (будет удалено после полного рефакторинга)
_state: Optional[AppState] = None


def _get_state() -> AppState:
    """Получить глобальное состояние (временная функция для миграции)."""
    global _state
    if _state is None:
        # Инициализация по умолчанию при первом обращении
        config_service = ConfigService()
        lighthouse_controller = UILighthouseController()
        _state = AppState(
            config_service=config_service,
            lighthouse_controller=lighthouse_controller,
        )
    return _state


def _handle_termination_signal(signum: int, _frame: Any) -> None:
    """
    Обработчик сигналов завершения процесса.

    По Ctrl+C (SIGINT) или аналогичным сигналам сразу ставим флаг остановки,
    чтобы разорвать бесконечные циклы стрима и фонового потока и не зависать
    на graceful shutdown uvicorn.
    """
    logger.info("Получен сигнал завершения %s, останавливаем фоновые потоки.", signum)
    _get_state().stop_event.set()


def _setup_signal_handlers() -> None:
    """Настроить обработчики сигналов для корректной остановки по Ctrl+C."""
    possible_signals = [
        getattr(signal, "SIGINT", None),
        getattr(signal, "SIGTERM", None),
        getattr(signal, "SIGBREAK", None),  # Windows-специфичный сигнал
    ]

    for sig in possible_signals:
        if sig is None:
            continue
        try:
            signal.signal(sig, _handle_termination_signal)
        except (OSError, ValueError):
            # Может не сработать внутри потоков или в средах без поддержки сигналов.
            logger.debug("Не удалось установить обработчик для сигнала %s.", sig)


def _load_initial_config() -> None:
    """Загрузить конфигурацию с диска в память (через ConfigService)."""
    # ConfigService автоматически загружает конфигурацию при инициализации
    # Эта функция оставлена для обратной совместимости
    pass


def _save_config_to_disk(config: Dict[str, Any]) -> None:
    """Сохранить конфигурацию в config.json (через ConfigService)."""
    state = _get_state()
    state.config_service.update_config(config)
    state.config_service.save_to_disk()


def _update_frame(frame: np.ndarray) -> None:
    state = _get_state()
    with state.frame_lock:
        state.latest_frame = frame.copy()


def _update_lamp(is_exceeded: bool) -> None:
    state = _get_state()
    state.lighthouse_controller.set_state(is_exceeded)


def _get_current_config() -> Dict[str, Any]:
    state = _get_state()
    return state.config_service.get_config()


def _camera_worker() -> None:
    """
    Фоновый поток: читает кадры с камеры, выполняет детекцию и обновляет состояние.

    Для простоты дублирует часть логики display_video_stream, но без окон OpenCV.
    """
    state = _get_state()
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

            # Единый обработчик детекций и скоростей
            px_to_m_scale = float(config.get("px_to_m_scale", DEFAULT_PX_TO_M_SCALE))
            frame_processor = FrameProcessor(
                speed_limit_kmh=speed_limit_kmh,
                red_hold_seconds=red_hold_seconds,
                px_to_m_scale=px_to_m_scale,
                speed_func=compute_speed_kmh,
            )

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

                    # Используем актуальные значения визуализации и масштаба
                    # из конфига, чтобы переключение галочки и калибровка
                    # применялись без перезапуска фонового потока.
                    live_config = state.config_service.get_config()
                    live_detection_cfg = live_config.get("detection", {})
                    draw_boxes_live = bool(live_detection_cfg.get("draw_boxes", True))
                    px_to_m_scale_live = state.config_service.get_px_to_m_scale()
                    # Обновляем масштаб в процессоре, если он изменился
                    if px_to_m_scale_live != frame_processor._px_to_m_scale:
                        frame_processor._px_to_m_scale = px_to_m_scale_live

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
                            raw_detections = yolo_detector.detect(frame)
                        except Exception:  # noqa: BLE001
                            raw_detections = []
                            detection_enabled = False

                        # Обрабатываем детекции через единый процессор
                        processed = frame_processor.process(
                            frame=frame,
                            detections=raw_detections,
                            now=now,
                        )

                        # Обновляем состояние лампы на основе результата обработки
                        if processed.any_speed_exceeded:
                            is_exceeded_any = True

                        # Отрисовка боксов (если включена)
                        if draw_boxes_live:
                            for det in processed.detections:
                                x1, y1, x2, y2 = det.bbox
                                score = det.score
                                label_speed = det.speed_kmh

                                # Цвет рамки и текста
                                box_color = (0, 255, 0)
                                text_color = (0, 255, 0)
                                if det.is_over_limit:
                                    box_color = (0, 0, 255)
                                    text_color = (0, 0, 255)

                                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                                if label_speed is not None:
                                    speed_m_s = label_speed / 3.6
                                    label = f"person {score:.2f} | {speed_m_s:.2f} m\\s"
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
                    new_index = int(state.config_service.get_camera_config().get("index", camera_index))
                    if new_index != camera_index:
                        break

            finally:
                camera.release()

            # Небольшая пауза перед возможной следующей попыткой.
            time.sleep(0.1)

    finally:
        _update_lamp(False)


def _start_worker_if_needed() -> None:
    state = _get_state()
    if state.worker_thread is not None and state.worker_thread.is_alive():
        return

    state.stop_event.clear()
    worker = threading.Thread(target=_camera_worker, name="camera-worker", daemon=True)
    state.worker_thread = worker
    worker.start()


def create_app() -> FastAPI:
    """Создать и настроить экземпляр FastAPI-приложения."""
    app = FastAPI(title="LighthouseForCycles Web")

    # Настраиваем обработчики сигналов, чтобы по Ctrl+C корректно гасить фоновые потоки.
    _setup_signal_handlers()

    # Инициализируем состояние приложения с зависимостями
    global _state
    if _state is None:
        config_service = ConfigService()
        lighthouse_controller = UILighthouseController()
        _state = AppState(
            config_service=config_service,
            lighthouse_controller=lighthouse_controller,
        )
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
        state = _get_state()
        try:
            state.config_service.update_config(payload)
            state.config_service.save_to_disk()
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        return _get_current_config()

    @app.get("/api/lamp")
    async def get_lamp_state() -> Dict[str, str]:
        with state.lamp_lock:
            lamp = "red" if state.is_speed_exceeded else "green"
        return {"lamp": lamp}

    def frame_generator() -> Iterator[bytes]:
        state = _get_state()
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
        state = _get_state()
        state.stop_event.set()
        worker = state.worker_thread
        if worker is not None:
            worker.join(timeout=5.0)
        state.lighthouse_controller.cleanup()

    return app


app = create_app()
