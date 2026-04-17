# Railway Chatbot

Node.js chatbot API with:
- Neo4j for graph/query context
- A Python FastAPI service for ticket confirmation prediction

## Project Structure

- `src/` → Node.js API (`/api/chat`)
- `ticketstatus/` → Python prediction service (`/predict`)

## Prerequisites

- Node.js 18+
- Python 3.10+
- Neo4j 5+ (Desktop, Docker, or AuraDB)

## 1) Environment Setup

Create your local env file from template:

```bash
cp .env.example .env
```

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Then edit `.env` values.

## 2) Install Dependencies

From project root:

```bash
npm install
```

From `ticketstatus/`:

```bash
pip install -r requirements.txt
```

## 3) Start Required Services

### A. Start Neo4j

You have 3 options:

1. **Local Neo4j (recommended for each developer)**
   - Run Neo4j on your own machine
   - Set `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` in `.env`

2. **Neo4j AuraDB (cloud)**
   - Create an Aura instance
   - Use Aura connection details in `.env`

3. **Use someone else's laptop Neo4j (not recommended)**
   - Their Neo4j must be network-accessible
   - Firewall + bind address + credentials must allow remote connections
   - Better to use AuraDB or each person running local Neo4j

### B. Start Python Prediction API

In `ticketstatus/`:

```bash
uvicorn predict_api:app --host 127.0.0.1 --port 8000 --reload
```

### C. Start Node API

In project root:

```bash
npm start
```

Node API runs on `http://localhost:3000` by default.

## 4) Test API

POST request:

```http
POST http://localhost:3000/api/chat
Content-Type: application/json
```

Body example:

```json
{
  "message": "Will my train ticket get confirmed?"
}
```

## Notes for Team Members

- This project needs **both** services running:
  - Node API (`npm start`)
  - Python prediction API (`uvicorn ...:8000`)
- If prediction service is not running, Node calls to `http://127.0.0.1:8000/predict` will fail.
- Never commit `.env` or API keys.

## Share Neo4j KG Snapshot (Recommended)

Use this when you want teammates to run the same graph data locally.

### 1) Export snapshot on your machine

Stop the Neo4j DB first, then run:

```bash
neo4j-admin database dump neo4j --to-path=./backup
```

This creates a dump file in `./backup` (for example `neo4j.dump`).

### 2) Send snapshot to teammate

- Zip and share the dump file (`neo4j.dump`) via Drive/GitHub Release/USB.
- Do not share credentials in chat; share separately.

### 3) Import snapshot on teammate machine

Stop their Neo4j DB, then run:

```bash
neo4j-admin database load neo4j --from-path=./backup --overwrite-destination=true
```

Start Neo4j again and confirm data in Neo4j Browser.

### 4) Configure this app to use imported DB

Update teammate `.env`:

```dotenv
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=their_local_password
```

### Docker note

If using Docker Neo4j, run `neo4j-admin` inside the container and mount a host backup folder.

### Version note

Use same major Neo4j version on both machines (for example 5.x to 5.x) to avoid dump/load compatibility issues.

## Troubleshooting

- **`Prediction failed`**
  - Check Python API is running on port `8000`
- **Neo4j auth/connection error**
  - Verify `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`
- **OpenAI auth error**
  - Verify `OPENAI_API_KEY` is valid and active
