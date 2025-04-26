const CACHE_NAME = 'face-clustering-cache-v1';
const STATIC_ASSETS = [
  '/static/css/bootstrap.min.css',
  '/static/js/jquery-3.5.1.slim.min.js',
  '/static/js/popper.min.js',
  '/static/js/bootstrap.min.js',
  '/static/fonts/fontawesome-all.min.css',
  // Add other static assets here
];

// Install event - cache static assets
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => {
        console.log('Caching static assets');
        return cache.addAll(STATIC_ASSETS);
      })
  );
});

// Activate event - clean up old caches
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames.filter(cacheName => {
          return cacheName.startsWith('face-clustering-cache-') &&
                 cacheName !== CACHE_NAME;
        }).map(cacheName => {
          return caches.delete(cacheName);
        })
      );
    })
  );
});

// Fetch event - serve from cache if available
self.addEventListener('fetch', event => {
  const requestUrl = new URL(event.request.url);

  // Special handling for image resources
  if (requestUrl.pathname.startsWith('/static/faces/')) {
    event.respondWith(cacheFirst(event.request));
  }
  // Regular static assets
  else if (
    requestUrl.pathname.startsWith('/static/') &&
    (requestUrl.pathname.endsWith('.css') ||
     requestUrl.pathname.endsWith('.js') ||
     requestUrl.pathname.endsWith('.woff') ||
     requestUrl.pathname.endsWith('.woff2') ||
     requestUrl.pathname.endsWith('.ttf'))
  ) {
    event.respondWith(cacheFirst(event.request));
  }
  // For other requests use network first
  else {
    event.respondWith(networkFirst(event.request));
  }
});

// Cache-first strategy
async function cacheFirst(request) {
  try {
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
      return cachedResponse;
    }

    const networkResponse = await fetch(request);
    if (networkResponse && networkResponse.status === 200) {
      const cache = await caches.open(CACHE_NAME);
      cache.put(request, networkResponse.clone());
    }
    return networkResponse;
  } catch (error) {
    console.error('Cache first strategy failed:', error);
    return new Response('Network and cache error', { status: 500 });
  }
}

// Network-first strategy
async function networkFirst(request) {
  try {
    const networkResponse = await fetch(request);
    if (networkResponse && networkResponse.status === 200) {
      const cache = await caches.open(CACHE_NAME);
      cache.put(request, networkResponse.clone());
    }
    return networkResponse;
  } catch (error) {
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
      return cachedResponse;
    }
    console.error('Network first strategy failed:', error);
    return new Response('Network error', { status: 500 });
  }
}