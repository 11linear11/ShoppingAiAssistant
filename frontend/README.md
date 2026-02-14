# Frontend Documentation

## 1. Role in the System
The frontend is a React + Vite chat client that:
- sends messages to backend `POST /api/chat`
- keeps `session_id` stable across turns
- renders assistant text + product cards
- applies fallback parsing when model text contains JSON-like product blocks

Main file: `frontend/src/App.jsx`

## 2. Runtime URL Strategy
- Production container mode:
  - `API_BASE = ''`
  - requests go to relative `/api` and are proxied by nginx
- Dev mode:
  - `VITE_API_URL=http://localhost:${BACKEND_HOST_PORT:-8080}`

## 3. Response Handling Pipeline
1. send user message to `/api/chat`
2. read `response` + `products`
3. if `products` empty, try extracting products from response text
4. strip JSON blocks / detail blocks from human-readable text
5. render final chat bubble + products

## 4. Parsing/Normalization Features
Implemented in `App.jsx`:
- Persian/Arabic digit normalization
- tolerant JSON repair/parsing
- field alias mapping (`name`, `product_name`, `brand_name`, ...)
- de-duplication by `name|price|product_url`
- detail parser for key-value formatted responses

## 5. Known Integration Expectations
Backend response should follow `ChatResponse` schema:
- `success`
- `response`
- `session_id`
- `products[]`
- `metadata`

If backend returns malformed text-only output, frontend fallback parser still attempts structured rendering.

## 6. Local Frontend Development
```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build frontend backend
```

Or standalone frontend dev in `frontend/`:
```bash
npm install
npm run dev -- --host 0.0.0.0
```

## 7. Troubleshooting
- UI shows no response:
  - check browser network for `/api/chat` status
  - check backend container logs
- API 200 but products not rendered:
  - inspect `products` field in response payload
  - inspect response text format and parser fallback behavior
