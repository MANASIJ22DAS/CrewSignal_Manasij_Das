const healthForm = document.getElementById('healthForm')
const connectivityStatus = document.getElementById('connectivityStatus')
const percentChanges = document.getElementById('percentChanges')
const ruleOutput = document.getElementById('ruleOutput')
const modelOutput = document.getElementById('modelOutput')
const alertAudio = document.getElementById('alertAudio')
const metricsArea = document.getElementById('metricsArea')
const sessionStartEl = document.getElementById('sessionStart')
const elapsedTimeEl = document.getElementById('elapsedTime')

const sessionStart = new Date()
const sessionStartIso = sessionStart.toISOString()
let sessionStartTime = sessionStart.getTime()

function readForm() {
  return {
    astronaut_id: document.getElementById('astronaut_id').value,
    astronaut_name: document.getElementById('astronaut_name').value,
    age: Number(document.getElementById('age').value),
    sex: document.getElementById('sex').value,
    time: document.getElementById('time').value || new Date().toLocaleString(),
    pulse: Number(document.getElementById('pulse').value),
    body_temperature: Number(document.getElementById('body_temperature').value),
    SpO2: Number(document.getElementById('SpO2').value),
    session_start: sessionStartIso
  }
}

function formatElapsed(ms) {
  const totalSeconds = Math.max(0, Math.floor(ms / 1000))
  const hh = String(Math.floor(totalSeconds / 3600)).padStart(2, '0')
  const mm = String(Math.floor((totalSeconds % 3600) / 60)).padStart(2, '0')
  const ss = String(totalSeconds % 60).padStart(2, '0')
  return `${hh}:${mm}:${ss}`
}

function updateElapsedTimer() {
  const elapsedMs = Date.now() - sessionStartTime
  elapsedTimeEl.textContent = formatElapsed(elapsedMs)
}

function calculatePercentChange(current, previous, key) {
  if (!previous || !Number.isFinite(Number(previous[key])) || Number(previous[key]) === 0) {
    return 0
  }
  return ((Number(current[key]) - Number(previous[key])) / Number(previous[key])) * 100
}

function saveToLocal(entry) {
  const current = JSON.parse(localStorage.getItem('crewSignalEntries') || '[]')
  const previous = current[current.length - 1] || null
  const prepared = { ...entry }

  if (previous) {
    prepared.pulse_change = calculatePercentChange(prepared, previous, 'pulse')
    prepared.body_temperature_change = calculatePercentChange(prepared, previous, 'body_temperature')
    prepared.spo2_change = calculatePercentChange(prepared, previous, 'SpO2')
  } else {
    prepared.pulse_change = 0
    prepared.body_temperature_change = 0
    prepared.spo2_change = 0
  }

  current.push(prepared)
  localStorage.setItem('crewSignalEntries', JSON.stringify(current))
  updateCharts(current)
  updateChangeSummary(current)
}

function updateCharts(entries) {
  const labels = entries.map((_, index) => index + 1)
  const pulse = entries.map(item => Number(item.pulse))
  const temperature = entries.map(item => Number(item.body_temperature))
  const spo2 = entries.map(item => Number(item.SpO2))

  Plotly.newPlot('heartRateChart', [{
    type: 'scatter',
    mode: 'lines',
    name: 'Pulse',
    x: labels,
    y: pulse,
    line: { color: '#ff5d73', width: 3 }
  }], {
    margin: { t: 20, r: 20, b: 40, l: 50 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#ecf8ff' },
    xaxis: { title: 'Reading' },
    yaxis: { title: 'Pulse' }
  }, { displayModeBar: false })

  Plotly.newPlot('tempChart', [{
    type: 'scatter',
    mode: 'lines',
    name: 'Body Temperature',
    x: labels,
    y: temperature,
    line: { color: '#1fb6ff', width: 3 }
  }], {
    margin: { t: 20, r: 20, b: 40, l: 50 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#ecf8ff' },
    xaxis: { title: 'Reading' },
    yaxis: { title: 'Temperature (°C)' }
  }, { displayModeBar: false })

  Plotly.newPlot('spo2Chart', [{
    type: 'scatter',
    mode: 'lines',
    name: 'SpO2',
    x: labels,
    y: spo2,
    line: { color: '#64d99b', width: 3 }
  }], {
    margin: { t: 20, r: 20, b: 40, l: 50 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#ecf8ff' },
    xaxis: { title: 'Reading' },
    yaxis: { title: 'SpO2' }
  }, { displayModeBar: false })
}

function updateChangeSummary(entries) {
  if (entries.length < 2) {
    percentChanges.textContent = 'No changes yet'
    return
  }

  const prev = entries[entries.length - 2]
  const curr = entries[entries.length - 1]
  const pulseDelta = calculatePercentChange(curr, prev, 'pulse')
  const tempDelta = calculatePercentChange(curr, prev, 'body_temperature')
  const spo2Delta = calculatePercentChange(curr, prev, 'SpO2')

  percentChanges.textContent = `Change since last entry: Pulse ${pulseDelta.toFixed(2)}%, Body Temperature ${tempDelta.toFixed(2)}%, SpO2 ${spo2Delta.toFixed(2)}%`
}

function updateConnectivityStatus() {
  const isOnline = navigator.onLine
  const localEntries = JSON.parse(localStorage.getItem('crewSignalEntries') || '[]').length
  const timeout = setTimeout(() => {
    connectivityStatus.textContent = isOnline ? 'Server timeout • local cache active' : 'Offline • local cache active'
    connectivityStatus.style.color = '#ffb454'
  }, 2000)

  fetch('/api/healthcheck', { cache: 'no-store' })
    .then(response => response.json())
    .then(() => {
      clearTimeout(timeout)
      connectivityStatus.textContent = isOnline ? `Online • server active • local data: ${localEntries}` : `Offline • local data: ${localEntries}`
      connectivityStatus.style.color = isOnline ? '#64d99b' : '#ffb454'
    })
    .catch(() => {
      clearTimeout(timeout)
      connectivityStatus.textContent = isOnline ? `Online • server timeout • local data: ${localEntries}` : `Offline • local data: ${localEntries}`
      connectivityStatus.style.color = '#ffb454'
    })
}

function buildMetricCard(title, metrics) {
  const card = document.createElement('div')
  card.className = 'metric-card'

  const heading = document.createElement('h3')
  heading.textContent = title

  const list = document.createElement('ul')
  const entries = [
    ['Accuracy', metrics.accuracy],
    ['Precision', metrics.precision],
    ['Recall', metrics.recall],
    ['F1-Score', metrics.f1_score],
    ['CV Score', metrics.cv_score]
  ]

  entries.forEach(([label, value]) => {
    const item = document.createElement('li')
    item.textContent = `${label}: ${Number(value).toFixed(4)}`
    list.appendChild(item)
  })

  card.appendChild(heading)
  card.appendChild(list)
  return card
}

function renderModelMetrics(metrics) {
  metricsArea.innerHTML = ''
  metricsArea.appendChild(buildMetricCard('Random Forest', metrics.rf))
  metricsArea.appendChild(buildMetricCard('SVM', metrics.svm))

  const labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
  const rfValues = [metrics.rf.accuracy, metrics.rf.precision, metrics.rf.recall, metrics.rf.f1_score]
  const svmValues = [metrics.svm.accuracy, metrics.svm.precision, metrics.svm.recall, metrics.svm.f1_score]

  Plotly.newPlot('performanceChart', [
    { type: 'bar', name: 'Random Forest', x: labels, y: rfValues, marker: { color: '#ff5d73' } },
    { type: 'bar', name: 'SVM', x: labels, y: svmValues, marker: { color: '#1fb6ff' } }
  ], {
    margin: { t: 20, r: 20, b: 40, l: 50 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#ecf8ff' },
    barmode: 'group'
  }, { displayModeBar: false })

  Plotly.newPlot('rfCmChart', [{
    type: 'heatmap',
    z: metrics.rf.confusion_matrix,
    x: ['Pred 0', 'Pred 1'],
    y: ['Actual 0', 'Actual 1'],
    colorscale: 'Plasma'
  }], {
    margin: { t: 20, r: 20, b: 40, l: 50 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#ecf8ff' },
    title: 'Random Forest Confusion Matrix'
  }, { displayModeBar: false })

  Plotly.newPlot('svmCmChart', [{
    type: 'heatmap',
    z: metrics.svm.confusion_matrix,
    x: ['Pred 0', 'Pred 1'],
    y: ['Actual 0', 'Actual 1'],
    colorscale: 'Viridis'
  }], {
    margin: { t: 20, r: 20, b: 40, l: 50 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#ecf8ff' },
    title: 'SVM Confusion Matrix'
  }, { displayModeBar: false })
}

function handleTrain() {
  fetch('/api/train', { method: 'POST' })
    .then(response => response.json())
    .then(data => {
      if (data.metrics) {
        renderModelMetrics(data.metrics)
      }
    })
    .catch(() => {})
}

function handlePredict() {
  const entry = readForm()
  const current = JSON.parse(localStorage.getItem('crewSignalEntries') || '[]')
  const previous = current[current.length - 1] || null

  if (previous) {
    entry.pulse_change = calculatePercentChange(entry, previous, 'pulse')
    entry.body_temperature_change = calculatePercentChange(entry, previous, 'body_temperature')
    entry.spo2_change = calculatePercentChange(entry, previous, 'SpO2')
  } else {
    entry.pulse_change = 0
    entry.body_temperature_change = 0
    entry.spo2_change = 0
  }

  fetch('/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(entry)
  })
    .then(response => response.json())
    .then(data => {
      if (data.rule_based_risk) {
        ruleOutput.textContent = `Overall Risk Category: ${data.overall_risk_category} | Recommended Next Step: ${data.recommended_next_step}`
      }
      if (data.rf_predicted_health_status !== undefined && data.svm_predicted_health_status !== undefined) {
        modelOutput.textContent = `RF: ${data.rf_predicted_health_status} | SVM: ${data.svm_predicted_health_status} | Agreement: ${data.model_agreement_status} | RF Prob: ${data.random_forest_probability.join(', ')} | SVM Decision: ${data.svm_decision_function.map(value => Number(value).toFixed(3)).join(', ')}`
      }

      const worstLabel = Number(data.worst_condition_label)
      if (Number(data.rf_predicted_health_status) === worstLabel && Number(data.svm_predicted_health_status) === worstLabel) {
        alertAudio.play().catch(() => {})
      }

      saveToLocal(entry)
    })
    .catch(() => saveToLocal(entry))
}

function init() {
  sessionStartEl.textContent = sessionStart.toLocaleString()
  setInterval(updateElapsedTimer, 1000)

  const existing = JSON.parse(localStorage.getItem('crewSignalEntries') || '[]')
  if (existing.length > 0) {
    updateCharts(existing)
    updateChangeSummary(existing)
  }

  updateConnectivityStatus()
  setInterval(updateConnectivityStatus, 4000)
}

document.getElementById('trainBtn').addEventListener('click', handleTrain)
document.getElementById('predictBtn').addEventListener('click', handlePredict)
document.getElementById('saveLocalBtn').addEventListener('click', () => saveToLocal(readForm()))
window.addEventListener('load', init)
