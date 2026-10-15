const CACHE_NAME = 'crew-signal-v1'
const OFFLINE_RESOURCES = ['/', '/static/index.html', '/static/styles.css', '/static/app.js', '/emergency_voice.mp3']
self.addEventListener('install', event => {
  event.waitUntil(caches.open(CACHE_NAME).then(cache => cache.addAll(OFFLINE_RESOURCES)))
  self.skipWaiting()
})
self.addEventListener('activate', event => {
  event.waitUntil(caches.keys().then(keys => Promise.all(keys.filter(key => key !== CACHE_NAME).map(key => caches.delete(key)))))
  event.waitUntil(self.clients.claim())
})
self.addEventListener('fetch', event => {
  if (event.request.method !== 'GET') {
    return
  }
  event.respondWith(caches.match(event.request).then(response => {
    if (response) {
      return response
    }
    return fetch(event.request).then(networkResponse => {
      if (networkResponse && networkResponse.status === 200) {
        const copy = networkResponse.clone()
        caches.open(CACHE_NAME).then(cache => cache.put(event.request, copy))
      }
      return networkResponse
    }).catch(() => caches.match('/static/index.html'))
  }))
})
