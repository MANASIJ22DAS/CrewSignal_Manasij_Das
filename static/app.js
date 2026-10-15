const healthForm = document.getElementById('healthForm')
const connectivityStatus = document.getElementById('connectivityStatus')
const percentChanges = document.getElementById('percentChanges')
const ruleOutput = document.getElementById('ruleOutput')
const modelOutput = document.getElementById('modelOutput')
const alertAudio = document.getElementById('alertAudio')
const metricsArea = document.getElementById('metricsArea')

const trendState = {
  pulse: [],
  body_temperature: [],
  SpO2: [],
  labels: []
}

function ensureSessionStart() {
  let sessionStart = localStorage.getItem('crewSignalSessionStart')
  if (!sessionStart) {
    sessionStart = new Date().toISOString()
    localStorage.setItem('crewSignalSessionStart', sessionStart)
  }
  const el = document.getElementById('sessionStart')
  if (el) el.textContent = new Date(sessionStart).toLocaleString()
  if (!window._elapsedTimer) {
    window._elapsedTimer = setInterval(() => {
      const startVal = localStorage.getItem('crewSignalSessionStart')
      if (!startVal) return
      const start = new Date(startVal).getTime()
      const diff = Date.now() - start
      const h = String(Math.floor(diff / 3600000)).padStart(2, '0')
      const m = String(Math.floor((diff % 3600000) / 60000)).padStart(2, '0')
      const s = String(Math.floor((diff % 60000) / 1000)).padStart(2, '0')
      const elElapsed = document.getElementById('elapsedTime')
      if (elElapsed) elElapsed.textContent = `${h}:${m}:${s}`
    }, 1000)
  }
}

function saveMeasurement(entry) {
  ensureSessionStart()
}

function readForm() {
  const timeVal = document.getElementById('time').value
  return {
    astronaut_id: document.getElementById('astronaut_id').value,
    astronaut_name: document.getElementById('astronaut_name').value,
    age: document.getElementById('age').value,
    sex: document.getElementById('sex').value,
    time: timeVal && timeVal.trim() !== '' ? timeVal : new Date().toISOString(),
    pulse: Number(document.getElementById('pulse').value),
    body_temperature: Number(document.getElementById('body_temperature').value),
    SpO2: Number(document.getElementById('SpO2').value)
  }
}

function saveToLocal(entry) {
  const current = JSON.parse(localStorage.getItem('crewSignalEntries') || '[]')
  current.push(entry)
  localStorage.setItem('crewSignalEntries', JSON.stringify(current))
  saveMeasurement(entry)
  updateCharts(current)
  updateChangeSummary(current)
}

function updateCharts(entries) {
  const labels = entries.map((_, index) => index + 1)
  const pulse = entries.map(item => Number(item.pulse))
  const temperature = entries.map(item => Number(item.body_temperature))
  const spo2 = entries.map(item => Number(item.SpO2))

  trendState.pulse = pulse
  trendState.body_temperature = temperature
  trendState.SpO2 = spo2
  trendState.labels = labels

  Plotly.newPlot('heartRateChart', [{ type: 'scatter', mode: 'lines', name: 'Pulse', x: labels, y: pulse, line: { color: '#ff5d73', width: 3 } }], {
    margin: { t: 20, r: 20, b: 40, l: 50 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#ecf8ff' },
    xaxis: { title: 'Reading' },
    yaxis: { title: 'Pulse' }
  }, { displayModeBar: false })

  Plotly.newPlot('tempChart', [{ type: 'scatter', mode: 'lines', name: 'Body Temperature', x: labels, y: temperature, line: { color: '#1fb6ff', width: 3 } }], {
    margin: { t: 20, r: 20, b: 40, l: 50 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#ecf8ff' },
    xaxis: { title: 'Reading' },
    yaxis: { title: 'Temperature (°C)' }
  }, { displayModeBar: false })

  Plotly.newPlot('spo2Chart', [{ type: 'scatter', mode: 'lines', name: 'SpO2', x: labels, y: spo2, line: { color: '#64d99b', width: 3 } }], {
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
  const pulseDelta = ((Number(curr.pulse) - Number(prev.pulse)) / Number(prev.pulse)) * 100
  const tempDelta = ((Number(curr.body_temperature) - Number(prev.body_temperature)) / Number(prev.body_temperature)) * 100
  const spo2Delta = ((Number(curr.SpO2) - Number(prev.SpO2)) / Number(prev.SpO2)) * 100
  percentChanges.textContent = `Change since last entry: Pulse ${pulseDelta.toFixed(2)}%, Body Temperature ${tempDelta.toFixed(2)}%, SpO2 ${spo2Delta.toFixed(2)}%`
}

function updateConnectivityStatus() {
  const isOnline = navigator.onLine
  const timeout = setTimeout(() => {
    connectivityStatus.textContent = isOnline ? 'Server timeout / offline mode' : 'Offline mode'
    connectivityStatus.style.color = '#ffb454'
  }, 2000)

  fetch('/api/healthcheck', { cache: 'no-store' })
    .then(response => response.json())
    .then(() => {
      clearTimeout(timeout)
      const localEntries = JSON.parse(localStorage.getItem('crewSignalEntries') || '[]').length
      connectivityStatus.textContent = isOnline ? `Online • local data: ${localEntries}` : `Offline • local data: ${localEntries}`
      connectivityStatus.style.color = isOnline ? '#64d99b' : '#ffb454'
    })
    .catch(() => {
      clearTimeout(timeout)
      const localEntries = JSON.parse(localStorage.getItem('crewSignalEntries') || '[]').length
      connectivityStatus.textContent = isOnline ? `Online (server not responding) • ${localEntries} local` : `Offline • ${localEntries} local`
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
    margin: { t: 20, r: 20, b: 50, l: 50 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#ecf8ff' },
    barmode: 'group'
  }, { displayModeBar: false })

  const rfMatrix = metrics.rf.confusion_matrix
  const svmMatrix = metrics.svm.confusion_matrix

  Plotly.newPlot('rfCmChart', [{
    type: 'heatmap',
    z: rfMatrix,
    x: ['Predicted 0', 'Predicted 1'],
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
    z: svmMatrix,
    x: ['Predicted 0', 'Predicted 1'],
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
  fetch('/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(entry)
  })
    .then(response => response.json())
    .then(data => {
      if (data.rule_based_risk) {
        ruleOutput.textContent = `Overall Risk Category: ${data.rule_based_risk}`
      }
      if (data.rf_predicted_health_status !== undefined && data.svm_predicted_health_status !== undefined) {
        modelOutput.textContent = `RF: ${data.rf_predicted_health_status} | SVM: ${data.svm_predicted_health_status} | Agreement: ${data.model_agreement_status}`
      }
      if (data.rf_predicted_health_status === 1 && data.svm_predicted_health_status === 1) {
        alertAudio.play()
      }
      saveToLocal(entry)
    })
    .catch(() => saveToLocal(readForm()))
}

function init() {
  ensureSessionStart()
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
