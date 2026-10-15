// dashboard.js
// Loads outputs/results.json and images. Performs connectivity checks (navigator.onLine,
// HTTP fetch to results.json with timeout) and localStorage check. No external APIs.

function setText(id, txt) { const el = document.getElementById(id); if (el) el.textContent = txt; }

async function fetchWithTimeout(url, timeout = 3000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);
  try {
    const resp = await fetch(url, {signal: controller.signal, cache: "no-store"});
    clearTimeout(timer);
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    return await resp.json();
  } catch (err) {
    clearTimeout(timer);
    throw err;
  }
}

async function loadResults() {
  // Connectivity checks
  const online = navigator.onLine;
  setText("connectivity-status", online ? "online (navigator.onLine=true)" : "offline (navigator.onLine=false)");
  // Try to fetch results.json with timeout
  try {
    const results = await fetchWithTimeout("outputs/results.json", 3000);
    setText("connectivity-status", (online ? "online" : "offline") + " — results.json loaded");
    populateDashboard(results);
  } catch (err) {
    setText("connectivity-status", (online ? "online" : "offline") + " — failed to load outputs/results.json (" + err.message + ")");
    // still try to display localStorage cached copy
    try {
      const cached = localStorage.getItem("crew_results");
      if (cached) {
        populateDashboard(JSON.parse(cached));
        setText("connectivity-status", document.getElementById("connectivity-status").textContent + " — shown cached result from localStorage");
      } else {
        setText("connectivity-status", document.getElementById("connectivity-status").textContent + " — no cached results in localStorage");
      }
    } catch (e) { /* ignore */ }
  }
}

function populateDashboard(results) {
  // Cache to localStorage
  try { localStorage.setItem("crew_results", JSON.stringify(results)); } catch (e) {}

  const rule = results.rule_based || {};
  setText("rule-risk", rule.overall_risk_category || "-");
  setText("rule-step", rule.recommended_next_step || "-");
  const ml = results.ml_outputs || {};
  setText("rf-status", ml.rf_predicted_status === 1 ? "unhealthy" : "healthy");
  setText("svm-status", ml.svm_predicted_status === 1 ? "unhealthy" : "healthy");
  setText("agreement", (ml.model_agreement ? "YES" : "NO"));
  setText("rf-prob", ml.rf_probability_last ? ml.rf_probability_last.join(", ") : "-");
  setText("svm-prob", ml.svm_probability_last ? ml.svm_probability_last.join(", ") : "-");

  // Show performance numbers
  if (results.model_performance) {
    const rf = results.model_performance.random_forest || {};
    const svm = results.model_performance.svm || {};
    const perfHtml = `
      <div><strong>Random Forest</strong> — Acc:${rf.accuracy?.toFixed(3)||"-"}, Prec:${rf.precision?.toFixed(3)||"-"}, Rec:${rf.recall?.toFixed(3)||"-"}, F1:${rf.f1_score?.toFixed(3)||"-"}, CV:${rf.cv_score?.toFixed(3)||"-"}</div>
      <div><strong>SVM</strong> — Acc:${svm.accuracy?.toFixed(3)||"-"}, Prec:${svm.precision?.toFixed(3)||"-"}, Rec:${svm.recall?.toFixed(3)||"-"}, F1:${svm.f1_score?.toFixed(3)||"-"}, CV:${svm.cv_score?.toFixed(3)||"-"}</div>
    `;
    document.getElementById("perf-list").innerHTML = perfHtml;
  }

  const charts = results.charts || {};
  if (charts.heart_rate_trend) document.getElementById("hr-trend").src = charts.heart_rate_trend;
  if (charts.temperature_trend) document.getElementById("temp-trend").src = charts.temperature_trend;
  if (charts.spo2_trend) document.getElementById("spo2-trend").src = charts.spo2_trend;
  if (charts.model_performance_comparison) document.getElementById("perf-chart").src = charts.model_performance_comparison;
  if (charts.rf_confusion_matrix) document.getElementById("rf-conf").src = charts.rf_confusion_matrix;
  if (charts.svm_confusion_matrix) document.getElementById("svm-conf").src = charts.svm_confusion_matrix;
}

window.addEventListener("load", loadResults);
window.addEventListener("online", () => setText("connectivity-status", "online (navigator.onLine=true)"));
window.addEventListener("offline", () => setText("connectivity-status", "offline (navigator.onLine=false)"));
