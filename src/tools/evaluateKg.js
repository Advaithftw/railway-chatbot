import fs from "fs";
import readline from "readline";
import path from "path";
import { findStationPaths } from "../services/queryService.js";

const CSV_PATH = path.resolve(process.cwd(), "ticketstatus/railway_dataset.csv");

const DEFAULT_SAMPLE_SIZE = Number(process.env.EVAL_SAMPLE_SIZE || 200);
const DEFAULT_MAX_HOPS = Number(process.env.KG_EVAL_MAX_HOPS || 3);
const DEFAULT_TIMEOUT_MS = Number(process.env.KG_EVAL_TIMEOUT_MS || 2000);
const DEFAULT_PATH_LIMIT = Number(process.env.KG_EVAL_PATH_LIMIT || 3);
const DEFAULT_CONCURRENCY = Math.max(1, Number(process.env.KG_EVAL_CONCURRENCY || 8));

const round = (value, digits = 2) => {
  if (!Number.isFinite(value)) return 0;
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
};

const percent = (numerator, denominator) => {
  if (!denominator) return 0;
  return round((Number(numerator) / Number(denominator)) * 100, 2);
};

const getPercentile = (sortedValues, percentileValue) => {
  if (!sortedValues.length) return 0;
  const index = Math.ceil((percentileValue / 100) * sortedValues.length) - 1;
  const boundedIndex = Math.max(0, Math.min(sortedValues.length - 1, index));
  return sortedValues[boundedIndex];
};

const summarizeDistribution = (values = []) => {
  const numeric = values.filter((v) => Number.isFinite(v)).map(Number);
  if (!numeric.length) {
    return {
      min: 0,
      avg: 0,
      p50: 0,
      p90: 0,
      p95: 0,
      max: 0,
    };
  }

  const sorted = [...numeric].sort((a, b) => a - b);
  const sum = sorted.reduce((acc, value) => acc + value, 0);

  return {
    min: round(sorted[0], 2),
    avg: round(sum / sorted.length, 2),
    p50: round(getPercentile(sorted, 50), 2),
    p90: round(getPercentile(sorted, 90), 2),
    p95: round(getPercentile(sorted, 95), 2),
    max: round(sorted[sorted.length - 1], 2),
  };
};

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
  concurrency = DEFAULT_CONCURRENCY,
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
    console.log(`Evaluation settings: maxHops=${maxHops}, timeout=${queryTimeoutMs}ms, pathLimit=${pathLimit}, concurrency=${concurrency}`);
  }

  const routeRows = [];
  let skipped = 0;
  let skippedInvalid = 0;

  for (const r of rows) {
    const source = r["Source Station"] || r["Source"] || r.Source;
    const destination = r["Destination Station"] || r["Destination"] || r.Destination;

    if (!source || !destination) {
      skippedInvalid += 1;
      continue;
    }

    const normalizedSource = String(source).trim();
    const normalizedDestination = String(destination).trim();

    if (!normalizedSource || !normalizedDestination) {
      skippedInvalid += 1;
      continue;
    }

    if (normalizedSource.toLowerCase() === normalizedDestination.toLowerCase()) {
      skipped += 1;
      continue;
    }

    routeRows.push({ source: normalizedSource, destination: normalizedDestination });
  }

  let success = 0;
  let timedOut = 0;
  let failedErrors = 0;
  let noPathFound = 0;
  let directCount = 0;
  let multiHopCount = 0;
  const queryLatenciesMs = [];
  const successLatenciesMs = [];
  const hopsObserved = [];
  const uniqueEvaluatedPairs = new Set();
  const examples = { success: [], fail: [] };

  const safeConcurrency = Math.max(1, Math.min(32, Number(concurrency) || DEFAULT_CONCURRENCY));
  let nextIndex = 0;

  const worker = async () => {
    while (nextIndex < routeRows.length) {
      const current = nextIndex;
      nextIndex += 1;

      const { source, destination } = routeRows[current];
      uniqueEvaluatedPairs.add(`${source.toLowerCase()}::${destination.toLowerCase()}`);

      if (enableLogs) {
        console.log(`[${current + 1}/${routeRows.length}] checking ${source} -> ${destination}`);
      }

      const start = Date.now();
      try {
        const paths = await withTimeout(
          findStationPaths(source, destination, { maxHops, limit: pathLimit }),
          queryTimeoutMs,
          `${source} -> ${destination}`
        );

        const latency = Date.now() - start;
        queryLatenciesMs.push(latency);

        if (paths && paths.length > 0) {
          success += 1;
          successLatenciesMs.push(latency);

          const chosenPath = paths.reduce((best, currentPath) => {
            const bestHops = Number(best?.hops ?? (best?.stations?.length || 1) - 1);
            const currentHops = Number(currentPath?.hops ?? (currentPath?.stations?.length || 1) - 1);
            return currentHops < bestHops ? currentPath : best;
          }, paths[0]);

          const hops = Number(chosenPath?.hops ?? (chosenPath?.stations?.length || 1) - 1);
          hopsObserved.push(hops);

          if (hops > 1) multiHopCount += 1;
          else directCount += 1;

          if (examples.success.length < 5) {
            examples.success.push({
              source,
              destination,
              path: formatPath(chosenPath),
              hops,
              routeType: hops > 1 ? "multi-hop" : "direct",
              candidatesFound: paths.length,
              latencyMs: latency,
              relTypes: chosenPath.relTypes,
            });
          }
        } else {
          noPathFound += 1;
          if (examples.fail.length < 5) {
            examples.fail.push({ source, destination, reason: "no_path_found", latencyMs: latency });
          }
        }
      } catch (err) {
        const latency = Date.now() - start;
        queryLatenciesMs.push(latency);

        if (String(err?.message || "").includes("timed out")) timedOut += 1;
        else failedErrors += 1;

        if (enableLogs) {
          console.warn(`Error checking ${source} -> ${destination}:`, err.message);
        }

        if (examples.fail.length < 5) {
          examples.fail.push({
            source,
            destination,
            reason: String(err?.message || "unknown_error"),
            latencyMs: latency,
          });
        }
      }
    }
  };

  const workers = Array.from(
    { length: Math.min(safeConcurrency, Math.max(1, routeRows.length)) },
    () => worker()
  );
  await Promise.all(workers);

  const evaluated = routeRows.length;

  const coverage = percent(success, rows.length);
  const recallAtK = percent(success, evaluated);
  const missRate = percent(noPathFound, evaluated);
  const timeoutRate = percent(timedOut, evaluated);
  const errorRate = percent(failedErrors, evaluated);
  const directHitRate = percent(directCount, evaluated);
  const multiHopHitRate = percent(multiHopCount, evaluated);

  const latencyStats = summarizeDistribution(queryLatenciesMs);
  const successLatencyStats = summarizeDistribution(successLatenciesMs);
  const hopsStats = summarizeDistribution(hopsObserved);

  const result = {
    sampleSize: rows.length,
    evaluatedCount: evaluated,
    uniqueEvaluatedPairs: uniqueEvaluatedPairs.size,
    maxHops,
    timeoutMs: queryTimeoutMs,
    pathLimit,
    concurrency: safeConcurrency,
    coveragePercent: coverage,
    recallAtKPercent: recallAtK,
    missRatePercent: missRate,
    timeoutRatePercent: timeoutRate,
    errorRatePercent: errorRate,
    directHitRatePercent: directHitRate,
    multiHopHitRatePercent: multiHopHitRate,
    successCount: success,
    noPathCount: noPathFound,
    failedErrorCount: failedErrors,
    directCount,
    multiHopCount,
    skipped,
    skippedInvalid,
    timedOut,
    latencyMs: latencyStats,
    successLatencyMs: successLatencyStats,
    hops: hopsStats,
    metricsVersion: "v2",
    examples,
  };

  if (enableLogs) {
    console.log(`KG coverage on sample (${rows.length}): ${coverage}% (${success}/${rows.length})`);
    console.log(`Recall@${pathLimit}: ${recallAtK}% | Miss rate: ${missRate}% | Timeout rate: ${timeoutRate}%`);
    console.log(`Direct hit rate: ${directHitRate}% | Multi-hop hit rate: ${multiHopHitRate}%`);
    console.log("Latency(ms):", latencyStats);
    console.log("Hops:", hopsStats);
    console.log(`Skipped(same): ${skipped}, skipped(invalid): ${skippedInvalid}, timed out: ${timedOut}, errors: ${failedErrors}`);
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
