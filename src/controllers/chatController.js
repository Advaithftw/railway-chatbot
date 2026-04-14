import { generateCypher } from "../services/llmService.js";
import { runQuery, getRouteFeatures, getAvailableTrains } from "../services/queryService.js";
import { extractEntities } from "../services/extractionService.js";
import { buildFeatures } from "../services/featureBuilder.js";
import { getPrediction } from "../services/predictionService.js";

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

      // 🧠 Step 1: Extract structured entities using LLM
      const entities = await extractEntities(query);
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
        for (const train of availableTrains.slice(0, 10)) {
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

    return res.json({
      type: "graph",
      cypher,
      result: dbResult
    });

  } catch (err) {
    console.error("ERROR:", err);
    return res.status(500).json({
      error: err.message || "Something went wrong"
    });
  }
};