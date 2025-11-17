// Dashboard Script - Uses data from data.js
// All data loaded from data.js: DASHBOARD_STATS, equipmentAreas, equipmentAlerts, equipmentPredictions

let notifications = [];
let charts = {};
let currentPredictionsFilter = "all";
let currentPredictionsPage = 1;
const predictionsPerPage = 20;
let realtimeInterval = null;
let currentView = "analytics";

// Toast notification system
function showToast(message, type = "info") {
  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  toast.textContent = message;
  toast.style.cssText = `
    position: fixed;
    top: 5rem;
    right: 2rem;
    padding: 1rem 1.5rem;
    background: ${
      type === "success" ? "#10b981" : type === "error" ? "#ef4444" : "#3b82f6"
    };
    color: white;
    border-radius: 0.5rem;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    z-index: 10000;
    animation: slideIn 0.3s ease-out;
    font-weight: 500;
  `;
  document.body.appendChild(toast);
  setTimeout(() => {
    toast.style.animation = "slideOut 0.3s ease-out";
    setTimeout(() => toast.remove(), 300);
  }, 2000);
}

// Handle area select change
function handleAreaSelectChange(value) {
  if (value === "overview") {
    showOverview();
  } else {
    showDetail(value);
  }
}

// Initialize
document.addEventListener("DOMContentLoaded", function () {
  console.log("Initializing dashboard...");

  // Start real-time updates
  startRealtimeUpdates();

  // Update KPI values from real data
  if (typeof DASHBOARD_STATS !== "undefined") {
    const stats = DASHBOARD_STATS;
    document.getElementById("totalEquipment").textContent =
      stats.total_equipment || stats.totalEquipment || 0;
    document.getElementById("criticalAlerts").textContent =
      stats.critical_alerts || stats.criticalAlerts || 0;
    document.getElementById("predictedDowntime").textContent =
      stats.predicted_downtime || stats.predictedDowntime || "0h";
  } else {
    console.warn("DASHBOARD_STATS not defined");
  }

  initCharts();
  renderAreaList();

  // Initialize predictions section
  if (
    typeof equipmentPredictions !== "undefined" &&
    equipmentPredictions.length > 0
  ) {
    renderPredictions();
    initPredictionFilters();
  } else {
    console.warn("equipmentPredictions not defined or empty");
  }

  console.log("Dashboard initialized!");
});

// Real-time Updates
function startRealtimeUpdates() {
  updateRealtimeIndicator();

  // Update every 30 seconds
  realtimeInterval = setInterval(() => {
    updateRealtimeIndicator();
    updateChartsWithRealtime();
  }, 30000);
}

function updateRealtimeIndicator() {
  const indicator = document.getElementById("realtimeIndicator");
  const lastUpdate = document.getElementById("lastUpdate");

  if (indicator && lastUpdate) {
    indicator.textContent = "🟢";
    lastUpdate.textContent = new Date().toLocaleTimeString("vi-VN");
  }
}

function updateChartsWithRealtime() {
  // Simulate small data updates for demonstration
  if (charts.rulTimeSeries && charts.rulTimeSeries.data) {
    charts.rulTimeSeries.data.datasets.forEach((dataset) => {
      const lastValue = dataset.data[dataset.data.length - 1];
      const variation = (Math.random() - 0.5) * 5;
      dataset.data.push(Math.max(20, Math.min(200, lastValue + variation)));
      dataset.data.shift();
    });
    charts.rulTimeSeries.update("none");
  }

  console.log("Real-time updated:", new Date().toLocaleTimeString("vi-VN"));
}

// Dashboard View Switching
function switchDashboardView(view) {
  currentView = view;

  // Update tab buttons
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.classList.remove("active");
    if (btn.dataset.view === view) {
      btn.classList.add("active");
    }
  });

  // Hide all views
  document.querySelectorAll(".dashboard-view").forEach((v) => {
    v.classList.add("hidden");
  });

  // Show selected view
  const viewElement = document.getElementById(view + "View");
  if (viewElement) {
    viewElement.classList.remove("hidden");
  }

  const viewNames = {
    analytics: "📊 Analytics Dashboard",
    reports: "📧 Reports Center",
    models: "🤖 Model Performance",
  };

  notifications.unshift({
    message: `Switched to ${viewNames[view]}`,
    systemId: "dashboard-view",
    equipmentName: view,
    detailMessage: `Viewing ${viewNames[view]} section`,
    timestamp: new Date().toLocaleString("vi-VN"),
  });
  updateNotificationPanel();
}

function initCharts() {
  console.log("Initializing charts...");

  // Risk Distribution Chart - use real data if available
  const riskCtx = document.getElementById("riskChart");
  if (!riskCtx) {
    console.error("riskChart canvas not found");
    return;
  }

  let riskData = [15, 25, 35, 25];

  if (
    typeof DASHBOARD_STATS !== "undefined" &&
    DASHBOARD_STATS.risk_distribution
  ) {
    const dist = DASHBOARD_STATS.risk_distribution;
    riskData = [
      dist.Critical || dist.critical || 0,
      dist.High || dist.high || 0,
      dist.Moderate || dist.moderate || 0,
      dist.Low || dist.low || 0,
    ];
    console.log("Risk distribution data:", riskData);
  }

  try {
    charts.risk = new Chart(riskCtx.getContext("2d"), {
      type: "pie",
      data: {
        labels: ["Critical", "High", "Moderate", "Low"],
        datasets: [
          {
            data: riskData,
            backgroundColor: ["#ef4444", "#f97316", "#eab308", "#22c55e"],
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        plugins: {
          legend: {
            position: "bottom",
          },
        },
      },
    });
    console.log("Risk chart created successfully");
  } catch (e) {
    console.error("Error creating risk chart:", e);
  }

  // RUL Time Series Chart
  initRULTimeSeriesChart();

  // Efficiency by Type Chart
  initEfficiencyByTypeChart();

  // Alerts Timeline Chart
  initAlertsTimelineChart();

  // Confidence Distribution Chart
  initConfidenceDistributionChart();
}

function initRULTimeSeriesChart() {
  const ctx = document.getElementById("rulTimeSeriesChart");
  if (!ctx) return;

  // Generate sample time series data for different equipment types
  const days = 30;
  const labels = [];
  for (let i = days; i >= 0; i--) {
    const date = new Date();
    date.setDate(date.getDate() - i);
    labels.push(
      date.toLocaleDateString("vi-VN", { month: "short", day: "numeric" })
    );
  }

  const datasets = [];
  if (typeof equipmentAreas !== "undefined" && equipmentAreas.length > 0) {
    const colors = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ef4444"];
    equipmentAreas.slice(0, 5).forEach((area, idx) => {
      const data = [];
      let baseRUL = area.avgRUL || 100;
      for (let i = 0; i < labels.length; i++) {
        baseRUL += (Math.random() - 0.48) * 10;
        data.push(Math.max(20, Math.min(200, baseRUL)));
      }
      datasets.push({
        label: area.name,
        data: data,
        borderColor: colors[idx],
        backgroundColor: colors[idx] + "20",
        tension: 0.4,
        borderWidth: 2,
        pointRadius: 1,
      });
    });
  }

  charts.rulTimeSeries = new Chart(ctx.getContext("2d"), {
    type: "line",
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      interaction: {
        mode: "index",
        intersect: false,
      },
      scales: {
        y: {
          beginAtZero: true,
          title: {
            display: true,
            text: "RUL (hours)",
          },
        },
      },
      plugins: {
        legend: {
          position: "bottom",
        },
      },
    },
  });
}

function initEfficiencyByTypeChart() {
  const ctx = document.getElementById("efficiencyByTypeChart");
  if (!ctx) return;

  const labels = [];
  const data = [];
  const colors = [];

  if (
    typeof equipmentPredictions !== "undefined" &&
    equipmentPredictions.length > 0
  ) {
    const typeStats = {};
    equipmentPredictions.forEach((pred) => {
      const type = pred.equipment_type || "Unknown";
      if (!typeStats[type]) {
        typeStats[type] = { total: 0, count: 0 };
      }
      const eff = pred.predicted_efficiency || pred.efficiency || 0.8;
      typeStats[type].total += eff;
      typeStats[type].count += 1;
    });

    const colorMap = {
      Compressor: "#3b82f6",
      Turbine: "#f59e0b",
      Pipeline: "#8b5cf6",
      Bearing: "#10b981",
      Pump: "#ef4444",
    };

    Object.keys(typeStats).forEach((type) => {
      labels.push(type);
      data.push(
        ((typeStats[type].total / typeStats[type].count) * 100).toFixed(1)
      );
      colors.push(colorMap[type] || "#6b7280");
    });
  }

  charts.efficiencyByType = new Chart(ctx.getContext("2d"), {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Hiệu suất trung bình (%)",
          data,
          backgroundColor: colors,
          borderColor: colors.map((c) => c),
          borderWidth: 2,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      scales: {
        y: {
          beginAtZero: true,
          max: 100,
          title: {
            display: true,
            text: "Hiệu suất (%)",
          },
        },
      },
      plugins: {
        legend: {
          display: false,
        },
      },
    },
  });
}

function initAlertsTimelineChart() {
  const ctx = document.getElementById("alertsTimelineChart");
  if (!ctx) return;

  // Generate weekly alerts data
  const weeks = 12;
  const labels = [];
  for (let i = weeks; i >= 0; i--) {
    const date = new Date();
    date.setDate(date.getDate() - i * 7);
    labels.push("W" + Math.ceil(date.getDate() / 7));
  }

  const criticalData = labels.map(() => Math.floor(Math.random() * 5) + 1);
  const highData = labels.map(() => Math.floor(Math.random() * 10) + 3);
  const moderateData = labels.map(() => Math.floor(Math.random() * 15) + 5);

  charts.alertsTimeline = new Chart(ctx.getContext("2d"), {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Critical",
          data: criticalData,
          borderColor: "#ef4444",
          backgroundColor: "#ef444420",
          fill: true,
          tension: 0.4,
        },
        {
          label: "High",
          data: highData,
          borderColor: "#f97316",
          backgroundColor: "#f9731620",
          fill: true,
          tension: 0.4,
        },
        {
          label: "Moderate",
          data: moderateData,
          borderColor: "#eab308",
          backgroundColor: "#eab30820",
          fill: true,
          tension: 0.4,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      scales: {
        y: {
          beginAtZero: true,
          stacked: false,
          title: {
            display: true,
            text: "Số cảnh báo",
          },
        },
      },
      plugins: {
        legend: {
          position: "bottom",
        },
      },
    },
  });
}

function initConfidenceDistributionChart() {
  const ctx = document.getElementById("confidenceDistributionChart");
  if (!ctx) return;

  const ranges = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"];
  const data = [0, 0, 0, 0, 0];

  if (
    typeof equipmentPredictions !== "undefined" &&
    equipmentPredictions.length > 0
  ) {
    equipmentPredictions.forEach((pred) => {
      const conf = (pred.confidence || 0.85) * 100;
      const idx = Math.min(4, Math.floor(conf / 20));
      data[idx]++;
    });
  }

  charts.confidenceDistribution = new Chart(ctx.getContext("2d"), {
    type: "doughnut",
    data: {
      labels: ranges,
      datasets: [
        {
          data,
          backgroundColor: [
            "#ef4444",
            "#f97316",
            "#eab308",
            "#3b82f6",
            "#10b981",
          ],
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      plugins: {
        legend: {
          position: "bottom",
        },
      },
    },
  });
}

function renderAreaList() {
  const list = document.getElementById("areaList");
  if (!list) {
    console.error("areaList element not found");
    return;
  }

  // Use real equipment areas from data.js if available
  const areas =
    typeof equipmentAreas !== "undefined" && Array.isArray(equipmentAreas)
      ? equipmentAreas
      : [
          {
            id: "pipeline",
            name: "Hệ thống Pipeline",
            risk: "high",
            avgRUL: 45,
            downtime: 12,
          },
          {
            id: "bearing",
            name: "Hệ thống Bearing",
            risk: "moderate",
            avgRUL: 120,
            downtime: 6,
          },
          {
            id: "pump",
            name: "Hệ thống Pump",
            risk: "low",
            avgRUL: 200,
            downtime: 2,
          },
          {
            id: "turbine",
            name: "Hệ thống Turbine",
            risk: "moderate",
            avgRUL: 100,
            downtime: 8,
          },
          {
            id: "compressor",
            name: "Hệ thống Compressor",
            risk: "low",
            avgRUL: 180,
            downtime: 4,
          },
        ];

  console.log("Rendering area list with", areas.length, "areas");

  list.innerHTML = areas
    .map(
      (area) => `
        <div class="area-item" onclick="showDetail('${area.id}')">
            <div class="area-info">
                <div class="risk-indicator risk-${area.risk}"></div>
                <div>
                    <div style="font-weight: 500;">${area.name}</div>
                    <div style="font-size: 0.875rem; color: #6b7280;">RUL: ${area.avgRUL}h | DT: ${area.downtime}h</div>
                </div>
            </div>
            <span style="font-size: 0.875rem; color: #6b7280; text-transform: uppercase;">${area.risk}</span>
        </div>
    `
    )
    .join("");
}

function showOverview() {
  document.getElementById("overviewScreen").classList.remove("hidden");
  document.getElementById("detailScreen").classList.add("hidden");
  document.getElementById("areaSelect").value = "overview";

  // Add visual feedback
  const areaSelect = document.getElementById("areaSelect");
  areaSelect.classList.add("active");
  setTimeout(() => areaSelect.classList.remove("active"), 1000);

  showToast("📊 Xem tổng quan", "info");
}

function showDetail(areaId) {
  const areas =
    typeof equipmentAreas !== "undefined" && Array.isArray(equipmentAreas)
      ? equipmentAreas
      : [];
  const area = areas.find((a) => a.id === areaId);

  if (!area) {
    console.warn("Area not found:", areaId);
    return;
  }

  // Add visual feedback to areaSelect
  const areaSelect = document.getElementById("areaSelect");
  if (areaSelect) {
    areaSelect.classList.add("active");
    setTimeout(() => areaSelect.classList.remove("active"), 1000);
  }

  document.getElementById("overviewScreen").classList.add("hidden");
  document.getElementById("detailScreen").classList.remove("hidden");
  document.getElementById("areaSelect").value = areaId;

  // Show toast
  const areaNames = {
    pipeline: "🔧 Pipeline",
    bearing: "⚙️ Bearing",
    pump: "💧 Pump",
    turbine: "🌪️ Turbine",
    compressor: "🏭 Compressor",
  };
  showToast(`Xem: ${areaNames[areaId] || area.name}`, "info");

  // Update KPIs
  document.getElementById("detailRUL").textContent = area.avgRUL + "h";
  document.getElementById("detailDowntime").textContent = area.downtime + "h";
  document.getElementById("rulProgress").style.width =
    Math.min((area.avgRUL / 200) * 100, 100) + "%";

  // Render RUL Trend
  if (charts.rulTrend) {
    charts.rulTrend.destroy();
  }
  const rulCtx = document.getElementById("rulTrendChart");
  if (!rulCtx) return;

  const labels = [
    "Day 1",
    "Day 2",
    "Day 3",
    "Day 4",
    "Day 5",
    "Day 6",
    "Day 7",
  ];
  const rulValues = generateRULTrend(area.avgRUL);

  charts.rulTrend = new Chart(rulCtx.getContext("2d"), {
    type: "line",
    data: {
      labels: labels,
      datasets: [
        {
          label: "RUL (h)",
          data: rulValues,
          borderColor: "#3b82f6",
          backgroundColor: "rgba(59, 130, 246, 0.1)",
          tension: 0.4,
        },
        {
          label: "Ngưỡng Bảo trì",
          data: Array(labels.length).fill(48),
          borderColor: "#ef4444",
          borderDash: [5, 5],
          borderWidth: 2,
          pointRadius: 0,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      scales: {
        y: {
          beginAtZero: true,
        },
      },
    },
  });

  // Render alerts for this area
  const alerts =
    typeof equipmentAlerts !== "undefined" && equipmentAlerts[areaId]
      ? equipmentAlerts[areaId]
      : [];
  renderAlerts(alerts);
}

function generateRULTrend(currentRUL) {
  const trend = [];
  for (let i = 7; i >= 0; i--) {
    trend.push(currentRUL + i * 5 + Math.random() * 10);
  }
  return trend;
}

function renderAlerts(alerts) {
  const alertList = document.getElementById("alertList");
  if (!alertList) return;

  if (alerts.length === 0) {
    alertList.innerHTML = '<p style="color: #6b7280;">Không có cảnh báo</p>';
    return;
  }

  alertList.innerHTML = alerts
    .map(
      (alert) => `
        <div class="alert-item ${alert.condition}">
            <div class="alert-equipment">${alert.name}</div>
            <div class="alert-metrics">
                <div class="metric-box">
                    <div class="metric-label">RUL còn lại</div>
                    <div class="metric-value">${alert.rul}h</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Xác suất</div>
                    <div class="metric-value">${alert.probability}%</div>
                </div>
            </div>
            <div style="font-size: 0.75rem; opacity: 0.8;">${alert.timestamp}</div>
            <div class="notification-buttons">
                <button class="btn btn-scada" onclick="sendToSCADA('${alert.id}', '${alert.name}')">📡 SCADA</button>
                <button class="btn btn-mes" onclick="sendToMES('${alert.id}', '${alert.name}')">🏭 MES</button>
                <button class="btn btn-erp" onclick="sendToERP('${alert.id}', '${alert.name}')">💼 ERP</button>
            </div>
            <div class="status-messages" id="status-${alert.id}"></div>
        </div>
    `
    )
    .join("");
}

// Integration buttons - UI only (simulated)
function sendToSCADA(equipmentId, equipmentName) {
  showNotification(
    `📡 Đang gửi cảnh báo tới SCADA...`,
    equipmentId,
    equipmentName,
    "🔄 Đồng bộ với SCADA hệ thống điều khiển tự động"
  );
}

function sendToMES(equipmentId, equipmentName) {
  showNotification(
    `🏭 Đang gửi lệnh bảo trì tới MES...`,
    equipmentId,
    equipmentName,
    "📋 Tạo work order trong Manufacturing Execution System"
  );
}

function sendToERP(equipmentId, equipmentName) {
  showNotification(
    `💼 Đang gửi yêu cầu phụ tùng tới ERP...`,
    equipmentId,
    equipmentName,
    "🛒 Kiểm tra tồn kho và đặt hàng phụ tùng"
  );
}

function showNotification(message, systemId, equipmentName, detailMessage) {
  const statusDiv = document.getElementById(`status-${systemId}`);

  setTimeout(() => {
    const statusMsg = document.createElement("div");
    statusMsg.className = "status-message";
    statusMsg.innerHTML = `
                    <div>✅ ${message}</div>
                    <div style="font-size: 0.7rem; margin-top: 0.25rem;">
                        <span style="opacity: 0.8;">${detailMessage}</span><br/>
                        <span style="opacity: 0.8;">ID: ${systemId} | ${new Date().toLocaleTimeString()}</span>
                    </div>
                `;
    if (statusDiv) statusDiv.appendChild(statusMsg);

    // Add to notification panel
    notifications.unshift({
      message,
      systemId,
      equipmentName,
      detailMessage,
      timestamp: new Date().toLocaleString(),
    });
    updateNotificationPanel();
  }, 500);
}

function updateNotificationPanel() {
  const panel = document.getElementById("notificationPanel");
  const list = document.getElementById("notificationList");

  if (!list) return;

  list.innerHTML = notifications
    .slice(0, 10)
    .map(
      (notif) => `
                <div class="notification-item">
                    <div style="font-weight: 500; margin-bottom: 0.25rem;">${notif.message}</div>
                    <div style="opacity: 0.8;">${notif.detailMessage}</div>
                    <div style="opacity: 0.8;">Thiết bị: ${notif.equipmentName}</div>
                    <div style="opacity: 0.7; margin-top: 0.25rem;">${notif.timestamp}</div>
                </div>
            `
    )
    .join("");

  if (notifications.length > 0 && panel) {
    panel.classList.add("active");
  }
}

// ============================================================================
// PREDICTIONS SECTION
// ============================================================================

function renderPredictions(filter = "all", page = 1) {
  const tbody = document.getElementById("predictionsTableBody");
  if (!tbody) {
    console.error("predictionsTableBody not found");
    return;
  }

  if (
    typeof equipmentPredictions === "undefined" ||
    !equipmentPredictions.length
  ) {
    tbody.innerHTML =
      '<tr><td colspan="7" style="padding: 2rem; text-align: center; color: #6b7280;">Không có dữ liệu dự đoán. Vui lòng chạy generate_predictions.py</td></tr>';
    return;
  }

  // Filter predictions
  let filtered = equipmentPredictions;

  if (filter === "high-risk") {
    filtered = filtered.filter((p) => {
      if (p.anomaly_probability && p.anomaly_probability > 0.7) return true;
      if (p.predicted_health && p.predicted_health < 0.4) return true;
      return false;
    });
  } else if (filter !== "all") {
    filtered = filtered.filter(
      (p) =>
        p.equipment_type &&
        p.equipment_type.toLowerCase() === filter.toLowerCase()
    );
  }

  // Pagination
  const startIdx = (page - 1) * predictionsPerPage;
  const endIdx = startIdx + predictionsPerPage;
  const paginated = filtered.slice(startIdx, endIdx);

  // Render table rows
  tbody.innerHTML = paginated
    .map((pred) => {
      // Determine risk badge
      let riskColor = "#22c55e";
      let riskText = "Low";

      if (pred.anomaly_probability) {
        if (pred.anomaly_probability > 0.7) {
          riskColor = "#ef4444";
          riskText = "High";
        } else if (pred.anomaly_probability > 0.4) {
          riskColor = "#f59e0b";
          riskText = "Medium";
        }
      } else if (pred.predicted_health) {
        if (pred.predicted_health < 0.4) {
          riskColor = "#ef4444";
          riskText = "High";
        } else if (pred.predicted_health < 0.6) {
          riskColor = "#f59e0b";
          riskText = "Medium";
        }
      }

      const riskBadge = `<span style="background: ${riskColor}; color: white; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.75rem; font-weight: 500;">${riskText}</span>`;

      // Format columns
      const rul = pred.predicted_rul
        ? `${Math.round(pred.predicted_rul)} days`
        : pred.rul_display || "-";
      const efficiency = pred.predicted_efficiency
        ? `${(pred.predicted_efficiency * 100).toFixed(1)}%`
        : pred.efficiency_display || "-";
      const confidence = pred.confidence
        ? `${(pred.confidence * 100).toFixed(1)}%`
        : pred.confidence_pct || "-";

      return `
      <tr style="border-bottom: 1px solid #e5e7eb; transition: background 0.2s;" 
          onmouseover="this.style.background='#f9fafb'" 
          onmouseout="this.style.background='white'">
        <td style="padding: 0.75rem; font-weight: 500; color: #1f2937;">${
          pred.equipment_id
        }</td>
        <td style="padding: 0.75rem;">
          <span style="background: #dbeafe; color: #1e40af; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.75rem; font-weight: 500;">
            ${pred.equipment_type || "Unknown"}
          </span>
        </td>
        <td style="padding: 0.75rem; color: #6b7280; font-size: 0.75rem;">${
          pred.model || "N/A"
        }</td>
        <td style="padding: 0.75rem; font-weight: 500;">${rul}</td>
        <td style="padding: 0.75rem; font-weight: 500;">${efficiency}</td>
        <td style="padding: 0.75rem;">${riskBadge}</td>
        <td style="padding: 0.75rem;">
          <div style="display: flex; align-items: center; gap: 0.5rem;">
            <div style="flex: 1; background: #e5e7eb; border-radius: 0.25rem; height: 0.5rem; overflow: hidden;">
              <div style="background: #10b981; height: 100%; width: ${confidence};"></div>
            </div>
            <span style="font-size: 0.75rem; color: #6b7280; min-width: 3rem;">${confidence}</span>
          </div>
        </td>
      </tr>
    `;
    })
    .join("");

  // Render pagination
  const totalPages = Math.ceil(filtered.length / predictionsPerPage);
  const pagination = document.getElementById("predictionsPagination");

  if (!pagination) return;

  if (totalPages > 1) {
    pagination.innerHTML = `
      <button onclick="changePredictionsPage(${page - 1})" ${
      page === 1 ? "disabled" : ""
    } 
        style="padding: 0.5rem 1rem; border: 1px solid #d1d5db; background: white; border-radius: 0.375rem; cursor: pointer; margin: 0 0.25rem;">
        &laquo; Trước
      </button>
      <span style="padding: 0 1rem;">Trang ${page} / ${totalPages} (${
      filtered.length
    } thiết bị)</span>
      <button onclick="changePredictionsPage(${page + 1})" ${
      page === totalPages ? "disabled" : ""
    } 
        style="padding: 0.5rem 1rem; border: 1px solid #d1d5db; background: white; border-radius: 0.375rem; cursor: pointer; margin: 0 0.25rem;">
        Sau &raquo;
      </button>
    `;
  } else {
    pagination.innerHTML = `<span>Hiển thị ${filtered.length} thiết bị</span>`;
  }
}

function initPredictionFilters() {
  const buttons = document.querySelectorAll(".pred-filter-btn");
  buttons.forEach((btn) => {
    btn.addEventListener("click", function () {
      // Update active state
      buttons.forEach((b) => {
        b.classList.remove("active");
        const color =
          b.dataset.filter === "high-risk"
            ? "#ef4444"
            : b.dataset.filter === "compressor"
            ? "#3b82f6"
            : b.dataset.filter === "turbine"
            ? "#f59e0b"
            : b.dataset.filter === "pipeline"
            ? "#8b5cf6"
            : "#10b981";
        b.style.background = "white";
        b.style.color = color;
      });

      this.classList.add("active");
      const color =
        this.dataset.filter === "high-risk"
          ? "#ef4444"
          : this.dataset.filter === "compressor"
          ? "#3b82f6"
          : this.dataset.filter === "turbine"
          ? "#f59e0b"
          : this.dataset.filter === "pipeline"
          ? "#8b5cf6"
          : "#10b981";
      this.style.background = color;
      this.style.color = "white";

      // Render with filter
      currentPredictionsFilter = this.dataset.filter;
      currentPredictionsPage = 1;
      renderPredictions(currentPredictionsFilter, currentPredictionsPage);
    });
  });
}

function changePredictionsPage(newPage) {
  currentPredictionsPage = newPage;
  renderPredictions(currentPredictionsFilter, currentPredictionsPage);
}

// ============================================================================
// EQUIPMENT REPORT FUNCTIONS
// ============================================================================

let selectedEquipment = [];

function toggleEquipmentSelection(equipId) {
  const checkbox = document.getElementById(`equip-${equipId}`);
  const card = checkbox.closest(".department-card");

  if (checkbox.checked) {
    checkbox.checked = false;
    card.classList.remove("selected");
    selectedEquipment = selectedEquipment.filter((d) => d !== equipId);

    // Visual feedback
    card.style.transform = "scale(0.98)";
    setTimeout(() => (card.style.transform = ""), 100);
    showToast(`\u274c B\u1ecf ch\u1ecdn: ${getEquipmentName(equipId)}`, "info");
  } else {
    checkbox.checked = true;
    card.classList.add("selected");
    selectedEquipment.push(equipId);

    // Visual feedback with animation
    card.style.transform = "scale(1.02)";
    setTimeout(() => (card.style.transform = ""), 100);
    showToast(
      `\u2705 \u0110\u00e3 ch\u1ecdn: ${getEquipmentName(equipId)}`,
      "success"
    );
  }

  // Update selection counter
  updateSelectionCounter();
}

function getEquipmentName(equipId) {
  const names = {
    pipeline: "\ud83d\udd27 Pipeline",
    bearing: "\u2699\ufe0f Bearing",
    pump: "\ud83d\udca7 Pump",
    turbine: "\ud83c\udf2a\ufe0f Turbine",
    compressor: "\ud83c\udfed Compressor",
  };
  return names[equipId] || equipId;
}

function updateSelectionCounter() {
  const count = selectedEquipment.length;
  const sendBtn = document.querySelector(".btn-send-report");
  if (sendBtn) {
    sendBtn.innerHTML = `\ud83d\udce4 G\u1eedi B\u00e1o c\u00e1o${
      count > 0 ? ` (${count})` : ""
    }`;
  }
}

function sendReportToDepartments() {
  if (selectedEquipment.length === 0) {
    showToast("⚠️ Vui lòng chọn thiết bị", "error");

    // Highlight equipment cards
    document.querySelectorAll(".department-card").forEach((card) => {
      card.style.animation = "shake 0.3s";
      setTimeout(() => (card.style.animation = ""), 300);
    });
    return;
  }

  const statusPanel = document.getElementById("reportStatusPanel");
  const statusContent = document.getElementById("reportStatusContent");

  statusPanel.style.display = "block";
  statusContent.innerHTML =
    '<div style="text-align: center; padding: 1rem;"><div style="font-size: 1.5rem;">⏳</div><div>Đang chuẩn bị và gửi báo cáo...</div></div>';

  const equipNames = {
    pipeline: "🔧 Pipeline",
    bearing: "⚙️ Bearing",
    pump: "💧 Pump",
    turbine: "🌪️ Turbine",
    compressor: "🏭 Compressor",
  };

  setTimeout(() => {
    const reportItems = selectedEquipment
      .map((equip) => {
        const reportType = getReportTypeForEquipment(equip);
        return `
        <div class="report-status-item">
          <div style="display: flex; justify-content: space-between; align-items: start;">
            <div style="flex: 1;">
              <div style="font-weight: 600; margin-bottom: 0.25rem;">
                ${equipNames[equip]}
              </div>
              <div style="font-size: 0.875rem; color: #6b7280;">
                ${reportType}
              </div>
              <div style="font-size: 0.75rem; color: #10b981; margin-top: 0.25rem;">
                ✅ Đã gửi thành công - ${new Date().toLocaleTimeString("vi-VN")}
              </div>
            </div>
            <div style="font-size: 1.5rem;">✉️</div>
          </div>
        </div>
      `;
      })
      .join("");

    statusContent.innerHTML = `
      <div style="margin-bottom: 1rem;">
        <div style="font-weight: 600; color: #10b981; margin-bottom: 0.5rem;">
          ✅ Đã gửi ${selectedEquipment.length} báo cáo thành công
        </div>
        <div style="font-size: 0.875rem; color: #6b7280;">
          Thời gian: ${new Date().toLocaleString("vi-VN")}
        </div>
      </div>
      ${reportItems}
    `;

    // Show success toast
    showToast(
      `✅ Gửi ${selectedEquipment.length} báo cáo thành công!`,
      "success"
    );

    // Add to notifications
    notifications.unshift({
      message: `📧 Đã gửi báo cáo dashboard`,
      systemId: "report-system",
      equipmentName: `${selectedEquipment.length} thiết bị`,
      detailMessage: `Báo cáo tóm tắt dashboard đã được gửi cho ${selectedEquipment
        .map((d) => equipNames[d])
        .join(", ")}`,
      timestamp: new Date().toLocaleString("vi-VN"),
    });
    updateNotificationPanel();
  }, 1500);
}

function getReportTypeForEquipment(equip) {
  const reportTypes = {
    pipeline: "📊 Corrosion analysis, thickness loss, RUL prediction",
    bearing: "⚙️ Vibration monitoring, fault detection, health status",
    pump: "💧 Performance tracking, efficiency analysis, maintenance schedule",
    turbine: "🌪️ Efficiency analysis, operational metrics, downtime forecast",
    compressor: "🏭 RUL prediction, pressure monitoring, maintenance planning",
  };
  return reportTypes[equip] || "Dashboard report";
}

function scheduleReport() {
  const frequency = document.getElementById("reportFrequency").value;
  const frequencyText = {
    daily: "hàng ngày",
    weekly: "hàng tuần",
    monthly: "hàng tháng",
  };

  if (selectedEquipment.length === 0) {
    showToast("⚠️ Vui lòng chọn thiết bị", "error");

    // Highlight equipment cards
    document.querySelectorAll(".department-card").forEach((card) => {
      card.style.animation = "shake 0.3s";
      setTimeout(() => (card.style.animation = ""), 300);
    });
    return;
  }

  const statusPanel = document.getElementById("reportStatusPanel");
  const statusContent = document.getElementById("reportStatusContent");

  statusPanel.style.display = "block";
  statusContent.innerHTML = `
    <div class="report-status-item" style="border-left-color: #f59e0b;">
      <div style="display: flex; justify-content: space-between; align-items: start;">
        <div style="flex: 1;">
          <div style="font-weight: 600; margin-bottom: 0.25rem;">
            ⏰ Đã lên lịch gửi báo cáo ${frequencyText[frequency]}
          </div>
          <div style="font-size: 0.875rem; color: #6b7280; margin-bottom: 0.25rem;">
            Gửi tới: ${selectedEquipment.length} thiết bị
          </div>
          <div style="font-size: 0.875rem; color: #6b7280;">
            Lần gửi tiếp theo: ${getNextSendTime(frequency)}
          </div>
          <div style="font-size: 0.75rem; color: #10b981; margin-top: 0.5rem;">
            ✅ Đã thiết lập thành công
          </div>
        </div>
        <div style="font-size: 1.5rem;">📅</div>
      </div>
    </div>
  `;

  // Show success toast
  showToast(`⏰ Đã lên lịch ${frequencyText[frequency]}`, "success");

  // Add to notifications
  notifications.unshift({
    message: `⏰ Lên lịch gửi báo cáo ${frequencyText[frequency]}`,
    systemId: "schedule-system",
    equipmentName: `${selectedEquipment.length} thiết bị`,
    detailMessage: `Báo cáo sẽ được gửi tự động ${frequencyText[frequency]} tới các thiết bị đã chọn`,
    timestamp: new Date().toLocaleString("vi-VN"),
  });
  updateNotificationPanel();
}

function getNextSendTime(frequency) {
  const now = new Date();
  let next = new Date();

  switch (frequency) {
    case "daily":
      next.setDate(now.getDate() + 1);
      next.setHours(8, 0, 0, 0);
      break;
    case "weekly":
      next.setDate(now.getDate() + ((7 - now.getDay() + 1) % 7));
      next.setHours(8, 0, 0, 0);
      break;
    case "monthly":
      next.setMonth(now.getMonth() + 1);
      next.setDate(1);
      next.setHours(8, 0, 0, 0);
      break;
  }

  return next.toLocaleString("vi-VN", {
    weekday: "long",
    year: "numeric",
    month: "long",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function updateDashboardTimeRange(days) {
  console.log(`Updating dashboard for ${days} days range`);

  // Add visual feedback to dropdown
  const timeRangeSelect = document.getElementById("timeRange");
  timeRangeSelect.classList.add("active");
  setTimeout(() => timeRangeSelect.classList.remove("active"), 1000);

  // Update charts with new time range
  if (charts.rulTimeSeries) {
    charts.rulTimeSeries.destroy();
  }
  initRULTimeSeriesChart();

  notifications.unshift({
    message: `📊 Đã cập nhật dashboard`,
    systemId: "dashboard-system",
    equipmentName: "Time Range",
    detailMessage: `Hiển thị dữ liệu ${days} ngày gần nhất`,
    timestamp: new Date().toLocaleString("vi-VN"),
  });
  updateNotificationPanel();

  // Show toast notification
  showToast(`📅 Hiển thị ${days} ngày`, "success");
}

// Get next scheduled send time
