import express from "express";
import dotenv from "dotenv";
import chatRoutes from "./routes/chatRoutes.js";
import axios from "axios";
import { getSession } from "./config/db.js";

dotenv.config();

const app = express();

app.use(express.json());
app.use(express.static("public", {
  etag: false,
  lastModified: false,
  maxAge: 0,
  setHeaders: (res, filePath) => {
    if (filePath.endsWith(".html")) {
      res.setHeader("Cache-Control", "no-store");
    }
  }
}));

app.use((req, res, next) => {
  const start = Date.now();
  res.on("finish", () => {
    const elapsed = Date.now() - start;
    console.log(`${req.method} ${req.originalUrl} -> ${res.statusCode} (${elapsed}ms)`);
  });
  next();
});

app.get("/api/health", async (_req, res) => {
  const status = {
    api: "ok",
    neo4j: "down",
    predictionApi: "down",
    ollama: "down"
  };

  try {
    const session = getSession();
    await session.run("RETURN 1 AS ok");
    await session.close();
    status.neo4j = "ok";
  } catch {
    status.neo4j = "down";
  }

  try {
    await axios.get("http://127.0.0.1:8000/docs", { timeout: 1500 });
    status.predictionApi = "ok";
  } catch {
    status.predictionApi = "down";
  }

  try {
    await axios.get("http://127.0.0.1:11434", { timeout: 1500 });
    status.ollama = "ok";
  } catch {
    status.ollama = "down";
  }

  const ok = status.neo4j === "ok" && status.predictionApi === "ok";
  return res.status(ok ? 200 : 503).json(status);
});

app.use("/api", chatRoutes);

const PORT = process.env.PORT || 3000;

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});