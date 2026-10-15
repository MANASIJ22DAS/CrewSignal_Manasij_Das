const DB_NAME = 'CrewSignalDB'
const DB_VERSION = 1
let dbInstance = null

function openDatabase() {
  if (dbInstance) {
    return Promise.resolve(dbInstance)
  }
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION)
    request.onupgradeneeded = event => {
      const db = event.target.result
      if (!db.objectStoreNames.contains('profiles')) {
        db.createObjectStore('profiles', { keyPath: 'astronaut_id' })
      }
      if (!db.objectStoreNames.contains('measurements')) {
        db.createObjectStore('measurements', { keyPath: 'timestamp' })
      }
      if (!db.objectStoreNames.contains('alerts')) {
        db.createObjectStore('alerts', { autoIncrement: true })
      }
    }
    request.onsuccess = event => {
      dbInstance = event.target.result
      resolve(dbInstance)
    }
    request.onerror = event => reject(event.target.error)
  })
}

async function saveMeasurement(record) {
  const db = await openDatabase()
  return new Promise((resolve, reject) => {
    const tx = db.transaction('measurements', 'readwrite')
    const store = tx.objectStore('measurements')
    store.put(record)
    tx.oncomplete = () => resolve(true)
    tx.onerror = event => reject(event.target.error)
  })
}

async function getAllMeasurements() {
  const db = await openDatabase()
  return new Promise((resolve, reject) => {
    const tx = db.transaction('measurements', 'readonly')
    const req = tx.objectStore('measurements').getAll()
    req.onsuccess = () => resolve(req.result)
    req.onerror = () => reject(req.error)
  })
}
