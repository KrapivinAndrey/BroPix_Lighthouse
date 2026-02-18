import React, { useEffect, useState } from "react";

const API_BASE = "/api";

function formatTimestamp(timestamp) {
  const date = new Date(timestamp * 1000);
  return date.toLocaleString("ru-RU", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function StatisticsPage({ onBack }) {
  const [events, setEvents] = useState([]);
  const [statistics, setStatistics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [page, setPage] = useState(0);
  const [total, setTotal] = useState(0);
  const limit = 50;

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 5000); // Обновление каждые 5 секунд
    return () => clearInterval(interval);
  }, [page]);

  const loadData = async () => {
    try {
      setError(null);
      const offset = page * limit;

      // Загружаем события
      const eventsRes = await fetch(
        `${API_BASE}/events?limit=${limit}&offset=${offset}`
      );
      if (!eventsRes.ok) throw new Error(`HTTP ${eventsRes.status}`);
      const eventsData = await eventsRes.json();
      setEvents(eventsData.events || []);
      setTotal(eventsData.total || 0);

      // Загружаем статистику
      const statsRes = await fetch(`${API_BASE}/statistics`);
      if (!statsRes.ok) throw new Error(`HTTP ${statsRes.status}`);
      const statsData = await statsRes.json();
      setStatistics(statsData);

      setLoading(false);
    } catch (e) {
      setError(String(e));
      setLoading(false);
    }
  };

  const handleExport = async () => {
    try {
      const res = await fetch(`${API_BASE}/export/json`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `lighthouse_events_${new Date().toISOString().split("T")[0]}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
    } catch (e) {
      alert(`Ошибка экспорта: ${e.message}`);
    }
  };

  const totalPages = Math.ceil(total / limit);

  return (
    <div className="page-root">
      <div className="neon-frame page-shell">
        <header className="page-header">
          <h1 className="logo-text">Статистика нарушений</h1>
          <button className="back-button" onClick={onBack}>
            ← Назад
          </button>
        </header>

        <main className="page-main">
          {loading && !statistics ? (
            <div className="status-text">Загрузка данных…</div>
          ) : error ? (
            <div className="status-text status-error">Ошибка: {error}</div>
          ) : (
            <>
              {/* Блок статистики */}
              <section className="statistics-section">
                <div className="section-title">Общая статистика</div>
                <div className="statistics-grid neon-frame">
                  {statistics && (
                    <>
                      <div className="stat-item">
                        <div className="stat-label">Всего событий</div>
                        <div className="stat-value">{statistics.total_events}</div>
                      </div>
                      <div className="stat-item">
                        <div className="stat-label">Уникальных объектов</div>
                        <div className="stat-value">{statistics.total_objects}</div>
                      </div>
                      <div className="stat-item">
                        <div className="stat-label">Средняя скорость</div>
                        <div className="stat-value">
                          {statistics.avg_speed_kmh != null
                            ? statistics.avg_speed_kmh.toFixed(1)
                            : "0.0"}{" "}
                          км/ч
                        </div>
                      </div>
                      <div className="stat-item">
                        <div className="stat-label">Максимальная скорость</div>
                        <div className="stat-value">
                          {statistics.max_speed_kmh != null
                            ? statistics.max_speed_kmh.toFixed(1)
                            : "0.0"}{" "}
                          км/ч
                        </div>
                      </div>
                    </>
                  )}
                </div>
                <button
                  className="primary-button"
                  onClick={handleExport}
                  style={{ marginTop: "1rem" }}
                >
                  Экспорт в JSON
                </button>
              </section>

              {/* Таблица событий */}
              <section className="events-section">
                <div className="section-title">Последние события</div>
                <div className="events-table-container neon-frame">
                  {events.length === 0 ? (
                    <div className="status-text">События не найдены</div>
                  ) : (
                    <>
                      <table className="events-table">
                        <thead>
                          <tr>
                            <th>Время</th>
                            <th>Скорость (км/ч)</th>
                            <th>Лимит (км/ч)</th>
                            <th>Превышение</th>
                            <th>Объект ID</th>
                          </tr>
                        </thead>
                        <tbody>
                          {events.map((event) => {
                            const exceedance = event.speed_kmh - event.speed_limit_kmh;
                            return (
                              <tr key={event.id}>
                                <td>{formatTimestamp(event.timestamp)}</td>
                                <td>{event.speed_kmh.toFixed(1)}</td>
                                <td>{event.speed_limit_kmh.toFixed(1)}</td>
                                <td className={exceedance > 0 ? "exceedance-positive" : ""}>
                                  +{exceedance.toFixed(1)} км/ч
                                </td>
                                <td>{event.track_id ?? "—"}</td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>

                      {/* Пагинация */}
                      {totalPages > 1 && (
                        <div className="pagination">
                          <button
                            className="pagination-button"
                            onClick={() => setPage(Math.max(0, page - 1))}
                            disabled={page === 0}
                          >
                            ← Предыдущая
                          </button>
                          <span className="pagination-info">
                            Страница {page + 1} из {totalPages} (всего: {total})
                          </span>
                          <button
                            className="pagination-button"
                            onClick={() =>
                              setPage(Math.min(totalPages - 1, page + 1))
                            }
                            disabled={page >= totalPages - 1}
                          >
                            Следующая →
                          </button>
                        </div>
                      )}
                    </>
                  )}
                </div>
              </section>
            </>
          )}
        </main>
      </div>
    </div>
  );
}

export default StatisticsPage;
