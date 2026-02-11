import React, { useEffect, useRef, useState } from "react";

const API_BASE = "/api";

function useConfig() {
  const [config, setConfig] = useState(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API_BASE}/config`)
      .then((res) => res.json())
      .then((data) => {
        setConfig(data);
        setLoading(false);
      })
      .catch((err) => {
        setError(String(err));
        setLoading(false);
      });
  }, []);

  const saveConfig = async (partial) => {
    setSaving(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/config`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(partial)
      });
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }
      const data = await res.json();
      setConfig(data);
    } catch (e) {
      setError(String(e));
    } finally {
      setSaving(false);
    }
  };

  return { config, loading, saving, error, saveConfig };
}

function useLamp() {
  const [lamp, setLamp] = useState("green");

  useEffect(() => {
    let cancelled = false;

    const tick = async () => {
      try {
        const res = await fetch(`${API_BASE}/lamp`);
        if (!res.ok) return;
        const data = await res.json();
        if (!cancelled && data.lamp) {
          setLamp(data.lamp);
        }
      } catch {
        // ignore
      }
    };

    tick();
    const id = setInterval(tick, 400);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  return lamp;
}

export function App() {
  const { config, loading, saving, error, saveConfig } = useConfig();
  const lamp = useLamp();
  const videoRef = useRef(null);
  // Таймер для скрытия кружков калибровки после завершения
  const calibrationHideTimerRef = useRef(null);
  const [isCalibrating, setIsCalibrating] = useState(false);
  const [calibrationPoints, setCalibrationPoints] = useState([]);
  const [calibrationMessage, setCalibrationMessage] = useState("");

  const cameraIndex = config?.camera?.index ?? 0;
  const speedLimit = config?.speed_limit_kmh ?? 8.0;
  const drawBoxes = config?.detection?.draw_boxes ?? true;
  const pxToMScale = config?.px_to_m_scale ?? null;

  // Очистка таймера скрытия кружков при размонтировании компонента
  useEffect(() => {
    return () => {
      if (calibrationHideTimerRef.current) {
        clearTimeout(calibrationHideTimerRef.current);
      }
    };
  }, []);

  const handleCameraChange = (e) => {
    const value = Number(e.target.value);
    if (Number.isNaN(value)) return;
    saveConfig({ camera: { index: value } });
  };

  const handleSpeedChange = (e) => {
    const value = Number(e.target.value);
    if (Number.isNaN(value)) return;
    saveConfig({ speed_limit_kmh: value });
  };

  const handleDrawBoxesChange = (e) => {
    const value = e.target.checked;
    saveConfig({ detection: { draw_boxes: value } });
  };

  const handleStartCalibration = () => {
    // При новом запуске калибровки сбрасываем таймер скрытия кружков
    if (calibrationHideTimerRef.current) {
      clearTimeout(calibrationHideTimerRef.current);
      calibrationHideTimerRef.current = null;
    }
    setIsCalibrating(true);
    setCalibrationPoints([]);
    setCalibrationMessage("Кликните по двум точкам на расстоянии 1 метр.");
  };

  const handleCalibrationClick = (e) => {
    if (!isCalibrating) return;
    const img = videoRef.current;
    if (!img) return;

    const rect = img.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) {
      setCalibrationMessage("Не удалось определить размер изображения.");
      return;
    }

    const naturalWidth = img.naturalWidth || config?.camera?.width;
    const naturalHeight = img.naturalHeight || config?.camera?.height;
    if (!naturalWidth || !naturalHeight) {
      setCalibrationMessage("Не удалось определить разрешение кадра камеры.");
      return;
    }

    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;

    const normX = clickX / rect.width;
    const normY = clickY / rect.height;

    const newPoints = [...calibrationPoints, { normX, normY }];

    if (newPoints.length < 2) {
      setCalibrationPoints(newPoints);
      setCalibrationMessage("Выберите вторую точку на расстоянии 1 метр.");
      return;
    }

    const [p1, p2] = newPoints;
    const x1 = p1.normX * naturalWidth;
    const y1 = p1.normY * naturalHeight;
    const x2 = p2.normX * naturalWidth;
    const y2 = p2.normY * naturalHeight;

    const dx = x2 - x1;
    const dy = y2 - y1;
    const distPx = Math.hypot(dx, dy);

    if (!Number.isFinite(distPx) || distPx <= 0) {
      setCalibrationMessage("Слишком маленькое расстояние между точками.");
      setCalibrationPoints([]);
      setIsCalibrating(false);
      return;
    }

    const distanceMeters = 1.0;
    const scale = distanceMeters / distPx;

    saveConfig({ px_to_m_scale: scale });
    setCalibrationPoints(newPoints);
    setIsCalibrating(false);
    setCalibrationMessage(
      `Калибровка сохранена: ~${scale.toFixed(4)} м/пикс.`
    );

    // Оставляем кружки видимыми ещё 2 секунды после калибровки, затем скрываем
    if (calibrationHideTimerRef.current) {
      clearTimeout(calibrationHideTimerRef.current);
    }
    calibrationHideTimerRef.current = setTimeout(() => {
      setCalibrationPoints([]);
      calibrationHideTimerRef.current = null;
    }, 2000);
  };

  const lampClass = lamp === "red" ? "lamp lamp-red" : "lamp lamp-green";

  return (
    <div className="page-root">
      <div className="neon-frame page-shell">
        <header className="page-header">
          <h1 className="logo-text">Маяк для велосипедиста</h1>
          <span className="subtitle">Панель управления</span>
        </header>

        <main className="page-main">
          <section className="video-section">
            <div className="section-title">Видеопоток</div>
            <div className="video-frame neon-frame">
              <div className="video-inner">
                <img
                  ref={videoRef}
                  src="/stream"
                  alt="Камера Lighthouse"
                  className="video-element"
                />
                <div
                  className={
                    isCalibrating
                      ? "calibration-overlay calibration-overlay-active"
                      : "calibration-overlay"
                  }
                  onClick={handleCalibrationClick}
                >
                  {calibrationPoints.map((p, idx) => (
                    <div
                      key={idx}
                      className="calibration-point"
                      style={{
                        left: `${p.normX * 100}%`,
                        top: `${p.normY * 100}%`
                      }}
                    />
                  ))}
                </div>
              </div>
            </div>
          </section>

          <section className="side-panel">
            <div className="section-title">Настройки</div>
            <div className="controls neon-frame">
              {loading ? (
                <div className="status-text">Загрузка конфига…</div>
              ) : (
                <>
                  <label className="field">
                    <span className="field-label">Номер камеры</span>
                    <input
                      type="number"
                      className="field-input"
                      value={cameraIndex}
                      onChange={handleCameraChange}
                    />
                  </label>

                  <label className="field">
                    <span className="field-label">
                      Ограничение скорости (км/ч)
                    </span>
                    <input
                      type="number"
                      className="field-input"
                      value={speedLimit}
                      onChange={handleSpeedChange}
                    />
                  </label>

                  <label className="field field-checkbox">
                    <input
                      type="checkbox"
                      className="field-checkbox-input"
                      checked={drawBoxes}
                      onChange={handleDrawBoxesChange}
                    />
                    <span className="field-label">
                      Показывать рамки вокруг объектов
                    </span>
                  </label>

                  <button
                    type="button"
                    className="primary-button"
                    onClick={handleStartCalibration}
                    disabled={loading || saving}
                  >
                    {isCalibrating ? "Режим калибровки…" : "Калибровка"}
                  </button>

                  {pxToMScale != null && (
                    <div className="hint-text">
                      Текущий масштаб: {pxToMScale.toFixed(4)} м/пикс.
                    </div>
                  )}

                  {calibrationMessage && (
                    <div className="status-text">{calibrationMessage}</div>
                  )}

                  <div className="status-block">
                    {saving && (
                      <div className="status-text">Сохранение настроек…</div>
                    )}
                    {error && (
                      <div className="status-text status-error">
                        Ошибка: {error}
                      </div>
                    )}
                  </div>
                  <div className="hint-text">
                    Смена номера камеры может потребовать перезапуска потока.
                  </div>
                </>
              )}
            </div>

            <div className="section-title lamp-title">Маяк</div>
            <div className="lamp-container neon-frame">
              <div className={lampClass} />
              <div className="lamp-caption">
                {lamp === "red"
                  ? "Превышение скорости"
                  : "Скорость в пределах лимита"}
              </div>
            </div>
          </section>
        </main>
      </div>
    </div>
  );
}

