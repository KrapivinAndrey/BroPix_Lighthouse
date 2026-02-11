import React, { useEffect, useState } from "react";

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

  const cameraIndex = config?.camera?.index ?? 0;
  const speedLimit = config?.speed_limit_kmh ?? 8.0;

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

  const lampClass = lamp === "red" ? "lamp lamp-red" : "lamp lamp-green";

  return (
    <div className="page-root">
      <div className="neon-frame page-shell">
        <header className="page-header">
          <h1 className="logo-text">LighthouseForCycles</h1>
          <span className="subtitle">Neon Control Panel</span>
        </header>

        <main className="page-main">
          <section className="video-section">
            <div className="section-title">Видеопоток</div>
            <div className="video-frame neon-frame">
              <img
                src="/stream"
                alt="Камера Lighthouse"
                className="video-element"
              />
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

                  {saving && (
                    <div className="status-text">Сохранение настроек…</div>
                  )}
                  {error && (
                    <div className="status-text status-error">
                      Ошибка: {error}
                    </div>
                  )}
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

