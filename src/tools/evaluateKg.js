import fs from "fs";
import readline from "readline";
import path from "path";
import { findStationPaths } from "../services/queryService.js";

const CSV_PATH = path.resolve(process.cwd(), "ticketstatus/railway_dataset.csv");

const DEFAULT_SAMPLE_SIZE = Number(process.env.EVAL_SAMPLE_SIZE || 200);
const DEFAULT_MAX_HOPS = Number(process.env.KG_EVAL_MAX_HOPS || 3);
const DEFAULT_TIMEOUT_MS = Number(process.env.KG_EVAL_TIMEOUT_MS || 2000);
const DEFAULT_PATH_LIMIT = Number(process.env.KG_EVAL_PATH_LIMIT || 3);

const formatPath = (pathObj) => {
  const stations = Array.isArray(pathObj?.stations) ? pathObj.stations.filter(Boolean) : [];
  return stations.length ? stations.join(" -> ") : "(no station chain returned)";
};

const withTimeout = (promise, ms, label) => {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(`${label} timed out after ${ms}ms`)), ms);
  });

  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
};

export async function evaluateKnowledgeGraph({
  sampleSize = DEFAULT_SAMPLE_SIZE,
  maxHops = DEFAULT_MAX_HOPS,
  queryTimeoutMs = DEFAULT_TIMEOUT_MS,
  pathLimit = DEFAULT_PATH_LIMIT,
  enableLogs = false,
} = {}) {
  if (!fs.existsSync(CSV_PATH)) {
    throw new Error(`CSV file not found: ${CSV_PATH}`);
  }

  const fileStream = fs.createReadStream(CSV_PATH);
  const rl = readline.createInterface({ input: fileStream, crlfDelay: Infinity });

  let header = null;
  const rows = [];

  for await (const line of rl) {
    if (!header) {
      header = line.split(",").map((h) => h.trim());
      continue;
    }
    // naive CSV parsing (dataset uses simple values for station codes)
    const cols = line.split(",");
    const obj = {};
    for (let i = 0; i < header.length; i++) {
      obj[header[i]] = (cols[i] || "").trim();
    }
    rows.push(obj);
    if (rows.length >= sampleSize) break;
  }

  if (enableLogs) {
    console.log(`Loaded ${rows.length} sample rows from dataset for evaluation`);
    console.log(`Evaluation settings: maxHops=${maxHops}, timeout=${queryTimeoutMs}ms, pathLimit=${pathLimit}`);
  }

  let success = 0;
  let skipped = 0;
  let timedOut = 0;
  let directCount = 0;
  let multiHopCount = 0;
  const examples = { success: [], fail: [] };

  for (let i = 0; i < rows.length; i++) {
    const r = rows[i];
    const source = r["Source Station"] || r["Source"] || r.Source;
    const destination = r["Destination Station"] || r["Destination"] || r.Destination;
    if (!source || !destination) continue;
    if (String(source).trim().toLowerCase() === String(destination).trim().toLowerCase()) {
      skipped += 1;
      continue;
    }

    if (enableLogs) {
      console.log(`[${i + 1}/${rows.length}] checking ${source} -> ${destination}`);
    }

    try {
      const paths = await withTimeout(
        findStationPaths(source, destination, { maxHops, limit: pathLimit }),
        queryTimeoutMs,
        `${source} -> ${destination}`
      );
      if (paths && paths.length > 0) {
        success += 1;
        const multiHopPath = paths.find((p) => (p.stations?.length || 0) > 2);
        const directPath = paths.find((p) => (p.stations?.length || 0) <= 2);
        const chosenPath = multiHopPath || directPath || paths[0];

        if ((chosenPath.stations?.length || 0) > 2) {
          multiHopCount += 1;
        } else {
          directCount += 1;
        }

        if (examples.success.length < 5) {
          examples.success.push({
            source,
            destination,
            path: formatPath(chosenPath),
            hops: chosenPath.hops,
            routeType: (chosenPath.stations?.length || 0) > 2 ? "multi-hop" : "direct",
            relTypes: chosenPath.relTypes,
          });
        }
      } else {
        if (examples.fail.length < 5) examples.fail.push({ source, destination });
      }
    } catch (err) {
      if (String(err?.message || "").includes("timed out")) {
        timedOut += 1;
      }
      if (enableLogs) {
        console.warn(`Error checking ${source} -> ${destination}:`, err.message);
      }
      if (examples.fail.length < 5) examples.fail.push({ source, destination, error: err.message });
    }
  }

  const coverage = ((success / rows.length) * 100).toFixed(2);
  const result = {
    sampleSize: rows.length,
    maxHops,
    timeoutMs: queryTimeoutMs,
    pathLimit,
    coveragePercent: Number(coverage),
    successCount: success,
    directCount,
    multiHopCount,
    skipped,
    timedOut,
    examples,
  };

  if (enableLogs) {
    console.log(`KG coverage on sample (${rows.length}): ${coverage}% (${success}/${rows.length})`);
    console.log(`Direct paths: ${directCount}, Multi-hop paths: ${multiHopCount}`);
    console.log(`Skipped: ${skipped}, Timed out: ${timedOut}`);
    console.log("Examples (success):", examples.success);
    console.log("Examples (fail):", examples.fail);
  }

  return result;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  evaluateKnowledgeGraph({ enableLogs: true })
    .catch((err) => {
      console.error("Evaluation failed:", err);
      process.exit(1);
    });
}
