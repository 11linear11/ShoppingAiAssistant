# Frontend (Shopping AI Assistant)

This frontend is a React + Vite client for the Shopping AI Assistant backend.

## 1. Responsibilities
- Send user messages to backend `POST /api/chat`
- Maintain `session_id` continuity across turns
- Render assistant text + product results
- Handle fallback extraction when product JSON is embedded in text

## 2. Runtime Modes

### 2.1 Local Dev (Vite)
Use compose debug stack:
```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

### 2.2 Production Container
Frontend is built and served via nginx in `docker/frontend.Dockerfile`.

## 3. Backend URL Behavior
- Production compose uses relative `/api` through nginx/proxy
- Dev override uses:
  - `VITE_API_URL=http://localhost:${BACKEND_HOST_PORT:-8080}`

## 4. Main Files
- `frontend/src/App.jsx`: main chat UI + API integration + message/product rendering
- `frontend/src/main.jsx`: app bootstrap
- `frontend/src/index.css`: styling
- `frontend/vite.config.js`: Vite configuration

## 5. Common Frontend Troubleshooting
- If backend is reachable but UI fails:
  - check browser network tab for `/api/chat`
  - verify `VITE_API_URL` in dev mode
- If text contains raw JSON block:
  - inspect backend `response` and `products`
  - verify fallback parser behavior in `App.jsx`
