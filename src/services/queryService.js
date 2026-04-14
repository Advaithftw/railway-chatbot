import { getSession } from "../config/db.js";

const escapeForCypher = (value = "") =>
  String(value).replace(/\\/g, "\\\\").replace(/"/g, '\\"').trim();

const toSafeNumber = (value) => {
  if (value === null || value === undefined) return null;

  if (typeof value === "number" && Number.isFinite(value)) {
    return Math.trunc(value);
  }

  if (typeof value === "string") {
    const numeric = Number(value.trim());
    if (Number.isFinite(numeric)) return Math.trunc(numeric);
    const match = value.match(/\b(\d{1,10})\b/);
    if (match) return Number(match[1]);
  }

  if (typeof value === "object") {
    if (typeof value.toNumber === "function") {
      const numeric = value.toNumber();
      if (Number.isFinite(numeric)) return Math.trunc(numeric);
    }

    if (Number.isFinite(value.low)) {
      return Math.trunc(value.low);
    }
  }

  return null;
};

export const runQuery = async (cypher) => {
  if (!cypher.toLowerCase().includes("match")) {
    throw new Error("Only MATCH queries allowed");
  }

  const session = getSession();

  try {
    const result = await session.run(cypher);
    return result.records.map(r => r.toObject());
  } finally {
    await session.close();
  }
};

export const getAvailableTrains = async (source, destination) => {
  const safeSource = escapeForCypher(source);
  const safeDestination = escapeForCypher(destination);

  const trainsQuery = `
  MATCH (t:Train)-[:STOPS_AT]->(s1:Station),
        (t)-[:STOPS_AT]->(s2:Station)
  WHERE toLower(s1.name) CONTAINS toLower("${safeSource}")
    AND toLower(s2.name) CONTAINS toLower("${safeDestination}")
  RETURN DISTINCT
         coalesce(t.number, t.train_number, t.trainNo, t.no) AS number,
         coalesce(t.name, t.trainName, "Unknown") AS name,
      coalesce(t.type, t.trainType, "Express") AS type,
      id(t) AS graphId
  LIMIT 50
  `;

  try {
    const result = await runQuery(trainsQuery);
    const trains = result
      .map((row) => {
        const parsedNumber = toSafeNumber(row.number) ?? toSafeNumber(row.graphId);
        const name = String(row.name || "Unknown").trim();
        const nameNumberMatch = name.match(/\b(\d{4,6})\b/);
        const fallbackNumber = nameNumberMatch ? Number(nameNumberMatch[1]) : null;

        return {
          number: parsedNumber || fallbackNumber,
          name,
          type: String(row.type || "Express").trim()
        };
      })
      .filter((train) => Number.isFinite(train.number));

    if (trains.length > 0) {
      return trains;
    }
  } catch (err) {
    console.warn("getAvailableTrains query failed:", err.message);
  }

  console.warn("No train numbers found via KG for route", { source, destination });
  return [];
};

export const getRouteFeatures = async (source, destination) => {
  const safeSource = escapeForCypher(source);
  const safeDestination = escapeForCypher(destination);

  const featureQuery = `
  MATCH (t:Train)-[:STOPS_AT]->(s1:Station),
        (t)-[:STOPS_AT]->(s2:Station)
  WHERE toLower(s1.name) CONTAINS toLower("${safeSource}")
    AND toLower(s2.name) CONTAINS toLower("${safeDestination}")
  WITH DISTINCT t
  OPTIONAL MATCH (t)-[:STOPS_AT]->(stop:Station)
  WITH t, count(stop) AS stopCount
  RETURN count(*) AS trainCount,
         avg(toFloat(stopCount)) AS avgStops,
         avg(CASE WHEN t.distance IS NULL THEN null ELSE toFloat(t.distance) END) AS avgDistance,
         avg(CASE WHEN t.travelTime IS NULL THEN null ELSE toFloat(t.travelTime) END) AS avgTravelTime
  `;

  const result = await runQuery(featureQuery);
  const route = result[0] || {};

  const trainCount = Number(route.trainCount) || 0;
  const stops = Math.max(2, Math.round(Number(route.avgStops) || trainCount || 8));
  const distance = Math.round(Number(route.avgDistance) || stops * 70);
  const time = Number((Number(route.avgTravelTime) || distance / 55).toFixed(1));

  return {
    distance,
    stops,
    time,
    trainCount
  };
};