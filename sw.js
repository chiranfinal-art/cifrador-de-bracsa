/* Modo offline do Cifrador de Brácsa.
   Ao publicar uma versão nova, mude o número em VERSAO para os aparelhos baixarem tudo de novo. */
const VERSAO = "cifrador-v11";
const ARQUIVOS = [
  "./", "index.html", "manifest.webmanifest", "exemplo.m4a",
  "icons/icon-192.png", "icons/icon-512.png", "icons/icon-maskable-512.png",
  "icons/apple-touch-icon.png", "icons/favicon.png",
];
self.addEventListener("install", e => {
  e.waitUntil(caches.open(VERSAO).then(c => c.addAll(ARQUIVOS)).then(() => self.skipWaiting()));
});
self.addEventListener("activate", e => {
  e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => k !== VERSAO).map(k => caches.delete(k)))).then(() => self.clients.claim()));
});
self.addEventListener("fetch", e => {
  const req = e.request;
  if (req.method !== "GET") return;
  const url = new URL(req.url);
  // fontes do Google: usa a cópia guardada e atualiza quando houver internet
  if (url.hostname.endsWith("fonts.googleapis.com") || url.hostname.endsWith("fonts.gstatic.com")) {
    e.respondWith(caches.open(VERSAO).then(async c => {
      const guardada = await c.match(req);
      const rede = fetch(req).then(r => { c.put(req, r.clone()); return r; }).catch(() => guardada);
      return guardada || rede;
    }));
    return;
  }
  if (url.origin !== location.origin) return;
  // arquivos do app: primeiro a cópia local (funciona sem internet)
  e.respondWith(caches.match(req, {ignoreSearch: true}).then(r => r || fetch(req).then(resp => {
    if (resp.ok) { const cp = resp.clone(); caches.open(VERSAO).then(c => c.put(req, cp)); }
    return resp;
  }).catch(() => caches.match("index.html"))));
});
