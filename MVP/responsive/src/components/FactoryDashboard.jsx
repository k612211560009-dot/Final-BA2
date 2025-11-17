import React, { useState } from "react";
import {
  AlertTriangle,
  Activity,
  Clock,
  TrendingDown,
  Send,
  CheckCircle,
  Bell,
  Calendar,
  Filter,
} from "lucide-react";
import {
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

const FactoryDashboard = () => {
  const [selectedArea, setSelectedArea] = useState("overview");
  const [timeRange, setTimeRange] = useState("7days");
  const [notifications, setNotifications] = useState([]);
  const [connectionStatus, setConnectionStatus] = useState({});

  const equipmentAreas = [
    {
      id: "assembly",
      name: "Khu Lắp ráp",
      risk: "high",
      avgRUL: 45,
      downtime: 12,
    },
    {
      id: "packaging",
      name: "Dây chuyền Đóng gói",
      risk: "moderate",
      avgRUL: 120,
      downtime: 6,
    },
    {
      id: "compression",
      name: "Hệ thống Nén Khí",
      risk: "low",
      avgRUL: 200,
      downtime: 2,
    },
    {
      id: "lineA",
      name: "Dây chuyền A",
      risk: "critical",
      avgRUL: 24,
      downtime: 18,
    },
  ];

  const equipmentAlerts = {
    assembly: [
      {
        id: 1,
        name: "Máy bơm P-101",
        condition: "Critical",
        rul: 24,
        probability: 85,
        timestamp: "2025-11-10 08:30",
      },
      {
        id: 2,
        name: "Động cơ M-203",
        condition: "Moderate",
        rul: 72,
        probability: 65,
        timestamp: "2025-11-10 09:15",
      },
    ],
    packaging: [
      {
        id: 3,
        name: "Băng tải B-501",
        condition: "Moderate",
        rul: 96,
        probability: 55,
        timestamp: "2025-11-10 07:45",
      },
    ],
    compression: [
      {
        id: 4,
        name: "Máy nén C-301",
        condition: "Normal",
        rul: 180,
        probability: 25,
        timestamp: "2025-11-10 06:00",
      },
    ],
    lineA: [
      {
        id: 5,
        name: "Động cơ C-05",
        condition: "Critical",
        rul: 18,
        probability: 92,
        timestamp: "2025-11-10 10:00",
      },
      {
        id: 6,
        name: "Robot R-102",
        condition: "Critical",
        rul: 36,
        probability: 78,
        timestamp: "2025-11-10 09:30",
      },
    ],
  };

  const rulTrendData = [
    { day: "T2", rul: 150, threshold: 100 },
    { day: "T3", rul: 135, threshold: 100 },
    { day: "T4", rul: 118, threshold: 100 },
    { day: "T5", rul: 95, threshold: 100 },
    { day: "T6", rul: 72, threshold: 100 },
    { day: "T7", rul: 58, threshold: 100 },
    { day: "CN", rul: 45, threshold: 100 },
  ];

  const riskDistribution = [
    { name: "Critical", value: 15, color: "#ef4444" },
    { name: "High", value: 25, color: "#f97316" },
    { name: "Moderate", value: 35, color: "#eab308" },
    { name: "Low", value: 25, color: "#22c55e" },
  ];

  const getConditionColor = (condition) => {
    switch (condition) {
      case "Critical":
        return "bg-red-100 text-red-800 border-red-300";
      case "Moderate":
        return "bg-yellow-100 text-yellow-800 border-yellow-300";
      case "Normal":
        return "bg-green-100 text-green-800 border-green-300";
      default:
        return "bg-gray-100 text-gray-800 border-gray-300";
    }
  };

  const getRiskColor = (risk) => {
    switch (risk) {
      case "critical":
        return "bg-red-500";
      case "high":
        return "bg-orange-500";
      case "moderate":
        return "bg-yellow-500";
      case "low":
        return "bg-green-500";
      default:
        return "bg-gray-500";
    }
  };

  const sendNotification = (type, alert) => {
    const timestamp = new Date().toLocaleTimeString("vi-VN");
    let message = "";
    let systemId = "";

    switch (type) {
      case "scada":
        message = `Cảnh báo SCADA - ${alert.name}: Tình trạng ${alert.condition}`;
        systemId = `SCADA-${Date.now()}`;
        break;
      case "mes":
        message = `Lệnh Bảo trì tạo cho ${alert.name}`;
        systemId = `WO-2025-${String(Date.now()).slice(-3)}`;
        break;
      case "erp":
        message = `Cập nhật ERP - Downtime dự kiến: ${alert.name}`;
        systemId = `ERP-${Date.now()}`;
        break;
    }

    setConnectionStatus((prev) => ({
      ...prev,
      [type + alert.id]: { status: "sending", timestamp },
    }));

    setTimeout(() => {
      const newNotification = {
        id: Date.now(),
        type,
        message,
        systemId,
        status: "success",
        timestamp,
      };
      setNotifications((prev) => [newNotification, ...prev].slice(0, 10));
      setConnectionStatus((prev) => ({
        ...prev,
        [type + alert.id]: { status: "success", timestamp, systemId },
      }));
    }, 1500);
  };

  const OverviewScreen = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white p-6 rounded-lg shadow-lg border-l-4 border-blue-500">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 mb-1">Tổng số Thiết bị</p>
              <p className="text-3xl font-bold text-gray-800">48</p>
            </div>
            <Activity className="text-blue-500" size={40} />
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow-lg border-l-4 border-red-500">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 mb-1">Cảnh báo Critical</p>
              <p className="text-3xl font-bold text-red-600">7</p>
            </div>
            <AlertTriangle className="text-red-500" size={40} />
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow-lg border-l-4 border-orange-500">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 mb-1">Downtime Dự đoán</p>
              <p className="text-3xl font-bold text-orange-600">38h</p>
              <p className="text-xs text-gray-500">7 ngày tới</p>
            </div>
            <Clock className="text-orange-500" size={40} />
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white p-6 rounded-lg shadow-lg">
          <h3 className="text-lg font-semibold mb-4 text-gray-800">
            Phân bổ Rủi ro Toàn nhà máy
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={riskDistribution}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) =>
                  `${name} ${(percent * 100).toFixed(0)}%`
                }
                outerRadius={100}
                fill="#8884d8"
                dataKey="value"
              >
                {riskDistribution.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-white p-6 rounded-lg shadow-lg">
          <h3 className="text-lg font-semibold mb-4 text-gray-800">
            Rủi ro theo Khu vực
          </h3>
          <div className="space-y-4">
            {equipmentAreas.map((area) => (
              <div
                key={area.id}
                className="flex items-center justify-between p-3 bg-gray-50 rounded-lg hover:bg-gray-100 cursor-pointer"
                onClick={() => setSelectedArea(area.id)}
              >
                <div className="flex items-center gap-3">
                  <div
                    className={`w-3 h-3 rounded-full ${getRiskColor(
                      area.risk
                    )}`}
                  ></div>
                  <div>
                    <p className="font-medium text-gray-800">{area.name}</p>
                    <p className="text-sm text-gray-500">
                      RUL: {area.avgRUL}h | DT: {area.downtime}h
                    </p>
                  </div>
                </div>
                <span className="text-sm text-gray-600 uppercase">
                  {area.risk}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );

  const EquipmentDetailScreen = ({ areaId }) => {
    const area = equipmentAreas.find((a) => a.id === areaId);
    const alerts = equipmentAlerts[areaId] || [];

    return (
      <div className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-white p-6 rounded-lg shadow-lg">
            <div className="flex items-center gap-3 mb-4">
              <TrendingDown className="text-blue-500" size={32} />
              <div>
                <p className="text-sm text-gray-600">RUL Trung bình</p>
                <p className="text-4xl font-bold text-blue-600">
                  {area?.avgRUL}h
                </p>
              </div>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-blue-600 h-2 rounded-full"
                style={{
                  width: `${Math.min((area?.avgRUL / 200) * 100, 100)}%`,
                }}
              ></div>
            </div>
          </div>

          <div className="bg-white p-6 rounded-lg shadow-lg">
            <div className="flex items-center gap-3 mb-4">
              <Clock className="text-orange-500" size={32} />
              <div>
                <p className="text-sm text-gray-600">Downtime Dự đoán</p>
                <p className="text-4xl font-bold text-orange-600">
                  {area?.downtime}h
                </p>
                <p className="text-xs text-gray-500">7 ngày tới</p>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow-lg">
          <h3 className="text-lg font-semibold mb-4 text-gray-800">
            Xu hướng RUL vs Ngưỡng Bảo trì
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={rulTrendData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="day" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line
                type="monotone"
                dataKey="rul"
                stroke="#3b82f6"
                strokeWidth={2}
                name="RUL Dự đoán"
              />
              <Line
                type="monotone"
                dataKey="threshold"
                stroke="#ef4444"
                strokeWidth={2}
                strokeDasharray="5 5"
                name="Ngưỡng Bảo trì"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-white p-6 rounded-lg shadow-lg">
          <div className="flex items-center gap-2 mb-4">
            <Bell className="text-red-500" size={24} />
            <h3 className="text-lg font-semibold text-gray-800">
              Cảnh báo & Thông báo
            </h3>
          </div>

          <div className="space-y-4">
            {alerts.map((alert) => (
              <div
                key={alert.id}
                className={`border-2 rounded-lg p-4 ${getConditionColor(
                  alert.condition
                )}`}
              >
                <div className="flex justify-between items-start mb-3">
                  <div>
                    <h4 className="font-bold text-lg">{alert.name}</h4>
                    <span
                      className={`inline-block px-3 py-1 rounded-full text-xs font-semibold mt-2 ${getConditionColor(
                        alert.condition
                      )}`}
                    >
                      {alert.condition}
                    </span>
                  </div>
                  <div className="text-right">
                    <p className="text-sm text-gray-600">{alert.timestamp}</p>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4 mb-4">
                  <div className="bg-white bg-opacity-50 p-3 rounded">
                    <p className="text-xs text-gray-600 mb-1">RUL Dự đoán</p>
                    <p className="text-2xl font-bold">{alert.rul}h</p>
                  </div>
                  <div className="bg-white bg-opacity-50 p-3 rounded">
                    <p className="text-xs text-gray-600 mb-1">Xác suất Lỗi</p>
                    <p className="text-2xl font-bold">{alert.probability}%</p>
                  </div>
                </div>

                <div className="border-t-2 border-opacity-30 pt-4 space-y-3">
                  <p className="text-sm font-semibold mb-2">
                    Gửi thông báo đến:
                  </p>

                  <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
                    <button
                      onClick={() => sendNotification("scada", alert)}
                      className="flex items-center justify-center gap-2 bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors"
                    >
                      <Send size={16} />
                      SCADA/Vận hành
                    </button>

                    <button
                      onClick={() => sendNotification("mes", alert)}
                      className="flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors"
                    >
                      <Send size={16} />
                      MES/Bảo trì
                    </button>

                    <button
                      onClick={() => sendNotification("erp", alert)}
                      className="flex items-center justify-center gap-2 bg-purple-600 hover:bg-purple-700 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors"
                    >
                      <Send size={16} />
                      ERP/Kế hoạch
                    </button>
                  </div>

                  <div className="space-y-1 text-xs">
                    {connectionStatus["scada" + alert.id] && (
                      <div className="flex items-center gap-2 text-green-700 bg-green-50 p-2 rounded">
                        <CheckCircle size={14} />
                        <span>
                          Thông báo SCADA đã gửi: Thành công (
                          {connectionStatus["scada" + alert.id].timestamp})
                        </span>
                      </div>
                    )}
                    {connectionStatus["mes" + alert.id] && (
                      <div className="flex items-center gap-2 text-green-700 bg-green-50 p-2 rounded">
                        <CheckCircle size={14} />
                        <span>
                          Lệnh Bảo trì (ID:{" "}
                          {connectionStatus["mes" + alert.id].systemId}) đã tạo
                          trong MES: Thành công
                        </span>
                      </div>
                    )}
                    {connectionStatus["erp" + alert.id] && (
                      <div className="flex items-center gap-2 text-green-700 bg-green-50 p-2 rounded">
                        <CheckCircle size={14} />
                        <span>
                          Cập nhật ERP đã gửi: Thành công (
                          {connectionStatus["erp" + alert.id].timestamp})
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gray-100 p-6">
      <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
          <div>
            <h1 className="text-3xl font-bold text-gray-800 mb-2">
              Dashboard Dự đoán Downtime Nhà máy
            </h1>
            <p className="text-gray-600">
              Giám sát tình trạng thiết bị và dự đoán thời gian ngừng máy
            </p>
          </div>

          <div className="flex gap-3">
            <div className="flex items-center gap-2">
              <Calendar size={20} className="text-gray-600" />
              <select
                value={timeRange}
                onChange={(e) => setTimeRange(e.target.value)}
                className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
              >
                <option value="7days">7 ngày</option>
                <option value="30days">30 ngày</option>
                <option value="90days">90 ngày</option>
              </select>
            </div>

            <div className="flex items-center gap-2">
              <Filter size={20} className="text-gray-600" />
              <select
                value={selectedArea}
                onChange={(e) => setSelectedArea(e.target.value)}
                className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
              >
                <option value="overview">Tổng quan</option>
                {equipmentAreas.map((area) => (
                  <option key={area.id} value={area.id}>
                    {area.name}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>
      </div>

      {selectedArea === "overview" ? (
        <OverviewScreen />
      ) : (
        <EquipmentDetailScreen areaId={selectedArea} />
      )}

      {notifications.length > 0 && (
        <div className="fixed bottom-6 right-6 w-96 max-h-96 overflow-y-auto bg-white rounded-lg shadow-2xl p-4 border-2 border-gray-200">
          <h4 className="font-bold text-gray-800 mb-3 flex items-center gap-2">
            <Bell size={18} />
            Lịch sử Thông báo
          </h4>
          <div className="space-y-2">
            {notifications.map((notif) => (
              <div
                key={notif.id}
                className="bg-green-50 border border-green-200 rounded p-3 text-xs"
              >
                <div className="flex items-start gap-2">
                  <CheckCircle size={14} className="text-green-600 mt-0.5" />
                  <div className="flex-1">
                    <p className="font-medium text-green-800">
                      {notif.message}
                    </p>
                    <p className="text-gray-600 mt-1">ID: {notif.systemId}</p>
                    <p className="text-gray-500 mt-1">{notif.timestamp}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default FactoryDashboard;
