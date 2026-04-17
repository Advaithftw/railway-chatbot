import { generateCypher } from "../services/llmService.js";
import { runQuery, getRouteFeatures, getAvailableTrains, findStationPaths } from "../services/queryService.js";
import { extractEntitiesFast } from "../services/extractionService.js";
import { buildFeatures } from "../services/featureBuilder.js";
import { getPrediction, getModelMetrics } from "../services/predictionService.js";
import { evaluateKnowledgeGraph } from "../tools/evaluateKg.js";

const MULTI_HOP_MAX_HOPS = Math.max(1, Math.min(8, Number(process.env.MULTI_HOP_MAX_HOPS || 4)));
const MULTI_HOP_PATH_LIMIT = Math.max(1, Math.min(50, Number(process.env.MULTI_HOP_PATH_LIMIT || 10)));

const cleanRouteToken = (value = "") =>
  String(value)
    .replace(/\bstation\b/gi, " ")
    .replace(/[^a-z0-9\s_-]/gi, " ")
    .replace(/\s+/g, " ")
    .trim();

const extractRouteFromQueryFallback = (queryText = "") => {
  const text = String(queryText || "");
  const match = text.match(/from\s+(.+?)\s+to\s+(.+?)(?=\?|\.|,|$)/i);
  if (!match) return { source: null, destination: null };
  const source = cleanRouteToken(match[1]);
  const destination = cleanRouteToken(match[2]);
  return {
    source: source || null,
    destination: destination || null,
  };
};

const resolveRouteFromQuery = async (queryText = "") => {
  try {
    const entities = await extractEntitiesFast(queryText);
    const source = cleanRouteToken(entities?.source || "");
    const destination = cleanRouteToken(entities?.destination || "");

    if (
      source &&
      destination &&
      source.toLowerCase() !== "unknown" &&
      destination.toLowerCase() !== "unknown"
    ) {
      return { source, destination };
    }
  } catch (err) {
    console.warn("Entity-based route extraction failed:", err.message);
  }

  return extractRouteFromQueryFallback(queryText);
};

export const chatHandler = async (req, res) => {
  const query = req.body.query?.toLowerCase();

  if (!query) {
    return res.status(400).json({ error: "Query is required" });
  }

  try {
    // =========================================================
    // 🔥 1. PREDICTION FLOW (WL / confirmation queries)
    // =========================================================
    if (query.includes("wl") || query.includes("confirm")) {

      // 🧠 Step 1: Fast deterministic entity extraction for low latency
      const entities = await extractEntitiesFast(query);
      console.log("Entities:", entities);

      // 🧠 Step 2: Get KG-based route features
      const kgData = await getRouteFeatures(
        entities.source,
        entities.destination
      );
      console.log("KG Data:", kgData);

      let bestPrediction = null;

      // 🧠 Step 3: If train number is provided, predict for that train only
      if (entities.trainNumber) {
        const features = buildFeatures(entities, kgData, entities.trainNumber, "Express");
        console.log("Final Features:", features);

        bestPrediction = await getPrediction(features);
        console.log("Prediction:", bestPrediction);
      } else {
        // 🧠 Step 3b: No specific train, find the BEST train for this confirmation
        const availableTrains = await getAvailableTrains(
          entities.source,
          entities.destination
        );
        console.log("Available Trains:", availableTrains);

        if (availableTrains.length === 0) {
          throw new Error(
            "No trains found on this route to predict confirmation for"
          );
        }

        const predictions = [];

        // 🤖 Call model for each train and collect predictions
        for (const train of availableTrains.slice(0, 5)) {
          const trainType = train.type || train.name?.split(" ")[0] || "Express";
          const features = buildFeatures(entities, kgData, train.number, trainType);
          try {
            const pred = await getPrediction(features);
            predictions.push({ train, prediction: pred });
          } catch (err) {
            console.warn(`Prediction failed for train ${train.number}:`, err.message);
          }
        }

        // 🎯 Find best prediction (highest probability)
        bestPrediction = predictions.reduce(
          (best, current) => {
            const currentProb = current.prediction?.probability ?? 0;
            const bestProb = best?.prediction?.probability ?? 0;
            return currentProb > bestProb ? current : best;
          },
          predictions[0]
        )?.prediction;

        if (!bestPrediction) {
          throw new Error("Could not get prediction for any train on this route");
        }

        console.log("Best Prediction:", bestPrediction);
      }

      const rawProbability = bestPrediction?.probability ?? 0;
      const prob = Number(rawProbability);

      if (!Number.isFinite(prob)) {
        throw new Error("Prediction API did not return a valid probability");
      }

      // Extract train info from model response
      const train = bestPrediction?.train ?? null;
      const pnr = bestPrediction?.pnr ?? null;

      // Extract prediction status from model response
      const predictionStatus = bestPrediction?.prediction ?? null;

      // 🧠 Step 5: Convert probability → human message
      let message = predictionStatus || "";
      if (!predictionStatus) {
        if (prob > 0.75) {
          message = "High chance of confirmation";
        } else if (prob > 0.4) {
          message = "Moderate chance of confirmation";
        } else {
          message = "Low chance of confirmation";
        }
      }

      return res.json({
        type: "prediction",
        train,
        pnr,
        probability: Number(prob.toFixed(4)),
        prediction: predictionStatus,
        message,
        details: {
          waitlist: entities.waitlist,
          class: entities.class,
          source: entities.source,
          destination: entities.destination
        }
      });
    }

    // =========================================================
    // 🔥 2. KG QUERY FLOW (normal railway queries)
    // =========================================================

    // 🧠 Step 1: Generate Cypher using LLM
    const cypher = await generateCypher(query);
    console.log("Generated Cypher:", cypher);

    // 🧠 Step 2: Run query on Neo4j
    const dbResult = await runQuery(cypher);
    console.log("DB Result:", dbResult);

    // Route extraction for direct + multi-hop augmentation
    const route = await resolveRouteFromQuery(query);

    let directTrains = [];
    if (route.source && route.destination) {
      try {
        directTrains = await getAvailableTrains(route.source, route.destination);
      } catch (err) {
        console.warn("Direct train lookup failed:", err.message);
      }
    }

    let multiHop = [];
    if (route.source && route.destination) {
      try {
        multiHop = await findStationPaths(route.source, route.destination, {
          maxHops: MULTI_HOP_MAX_HOPS,
          limit: MULTI_HOP_PATH_LIMIT,
        });
      } catch (err) {
        console.warn("Multi-hop search failed:", err.message);
      }
    }

    const isRouteQuery = Boolean(route.source && route.destination);
    const directCount = isRouteQuery ? directTrains.length : (dbResult?.length || 0);
    const hasDirect = directCount > 0;
    const answer = hasDirect
      ? `Found ${directCount} direct train option(s).${multiHop.length ? ` Also found ${multiHop.length} multi-hop path(s).` : ""}`
      : multiHop.length
      ? `No direct trains found; discovered ${multiHop.length} multi-hop path(s).`
      : "No trains or multi-hop paths found for this route in current graph data.";

    return res.json({
      type: "graph",
      cypher,
      result: isRouteQuery ? [] : dbResult,
      directTrains,
      multiHop,
      route,
      answer
    });

  } catch (err) {
    console.error("ERROR:", err);
    return res.status(500).json({
      error: err.message || "Something went wrong"
    });
  }
};

export const mlEvaluationHandler = async (_req, res) => {
  try {
    const metrics = await getModelMetrics();
    return res.json({
      type: "ml-evaluation",
      ...metrics,
    });
  } catch (err) {
    return res.status(500).json({
      error: err.message || "Failed to fetch ML metrics"
    });
  }
};

export const kgEvaluationHandler = async (req, res) => {
  try {
    const sampleSize = Number(req.query.sampleSize || process.env.EVAL_SAMPLE_SIZE || 100);
    const maxHops = Number(req.query.maxHops || process.env.KG_EVAL_MAX_HOPS || 3);
    const timeoutMs = Number(req.query.timeoutMs || process.env.KG_EVAL_TIMEOUT_MS || 2000);
    const pathLimit = Number(req.query.pathLimit || process.env.KG_EVAL_PATH_LIMIT || 3);

    const evaluation = await evaluateKnowledgeGraph({
      sampleSize,
      maxHops,
      queryTimeoutMs: timeoutMs,
      pathLimit,
      enableLogs: false,
    });

    return res.json({
      type: "kg-evaluation",
      ...evaluation,
    });
  } catch (err) {
    return res.status(500).json({
      error: err.message || "Failed to evaluate KG"
    });
  }
};