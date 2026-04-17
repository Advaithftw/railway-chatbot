import { getSession } from "../config/db.js";

const ROUTE_CACHE_TTL_MS = Number(process.env.ROUTE_CACHE_TTL_MS || 5 * 60 * 1000);
const routeFeatureCache = new Map();
const stationConnectionCache = new Map();

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

export const runQuery = async (cypher, params = {}) => {
  if (!cypher.toLowerCase().includes("match")) {
    throw new Error("Only MATCH queries allowed");
  }

  const session = getSession();

  try {
    const result = await session.run(cypher, params);
    return result.records.map((r) => r.toObject());
  } finally {
    await session.close();
  }
};

const normalizeStationKey = (value) => String(value || "").trim();

const stationMatches = (value, target) => {
  const normalizedValue = String(value || "").trim().toLowerCase();
  const normalizedTarget = String(target || "").trim().toLowerCase();

  return Boolean(normalizedValue) && Boolean(normalizedTarget) && normalizedValue === normalizedTarget;
};

const getStationConnections = async (station, limit = 25) => {
  const safeStation = normalizeStationKey(station);
  const safeLimit = Math.max(1, Math.min(100, Math.trunc(Number(limit) || 25)));
  const cacheKey = `${safeStation.toLowerCase()}::${safeLimit}`;
  const cached = stationConnectionCache.get(cacheKey);

  if (cached && Date.now() - cached.ts < ROUTE_CACHE_TTL_MS) {
    return cached.data;
  }

  const neighborsQuery = `
  MATCH (s1:Station)<-[:STOPS_AT]-(t:Train)-[:STOPS_AT]->(s2:Station)
  WHERE any(v IN [s1.name, s1.code, s1.stationCode, s1.station_name, s1.id]
        WHERE toLower(trim(toString(coalesce(v, "")))) = toLower($station))
      AND NOT any(v IN [s2.name, s2.code, s2.stationCode, s2.station_name, s2.id]
      WHERE toLower(trim(toString(coalesce(v, "")))) = toLower($station))
  RETURN DISTINCT
         coalesce(s2.code, s2.name, s2.stationCode, s2.station_name, s2.id) AS neighbor,
         collect(DISTINCT coalesce(t.number, t.train_number, t.trainNo, t.no, t.name, "Unknown")) AS trains
  ORDER BY neighbor
    LIMIT ${safeLimit}
  `;

    const rows = await runQuery(neighborsQuery, { station: safeStation });

  const data = rows
    .map((row) => ({
      station: String(row.neighbor || "").trim(),
      trains: Array.isArray(row.trains) ? row.trains.filter(Boolean).map(String) : [],
    }))
    .filter((row) => row.station);

  stationConnectionCache.set(cacheKey, { ts: Date.now(), data });
  return data;
};

// Find multi-hop station chains between two stations using BFS over train connections.
export const findStationPaths = async (
  source,
  destination,
  { maxHops = 4, limit = 10 } = {}
) => {
  const safeSource = normalizeStationKey(source);
  const safeDestination = normalizeStationKey(destination);
  const safeMaxHops = Math.max(1, Math.min(8, Number(maxHops) || 4));
  const safeLimit = Math.max(1, Math.min(50, Number(limit) || 10));

  if (!safeSource || !safeDestination) return [];

  const queue = [{ station: safeSource, stations: [safeSource], trains: [] }];
  const visitedDepth = new Map([[safeSource.toLowerCase(), 0]]);
  const results = [];

  while (queue.length && results.length < safeLimit) {
    const current = queue.shift();
    const depth = current.stations.length - 1;

    if (depth >= safeMaxHops) continue;

    const connections = await getStationConnections(current.station, 50);

    for (const connection of connections) {
      const nextStation = connection.station;
      const nextKey = nextStation.toLowerCase();

      if (current.stations.some((s) => stationMatches(s, nextStation))) {
        continue;
      }

      const nextStations = [...current.stations, nextStation];
      const nextTrains = [...current.trains, connection.trains];
      const nextDepth = depth + 1;

      if (stationMatches(nextStation, safeDestination)) {
        results.push({
          stations: nextStations,
          relTypes: Array.from({ length: nextDepth }, () => "STOPS_AT"),
          hops: nextDepth,
          trains: nextTrains,
        });
        if (results.length >= safeLimit) break;
      }

      const seenDepth = visitedDepth.get(nextKey);
      if (nextDepth < safeMaxHops && (seenDepth === undefined || nextDepth < seenDepth)) {
        visitedDepth.set(nextKey, nextDepth);
        queue.push({ station: nextStation, stations: nextStations, trains: nextTrains });
      }
    }
  }

  return results;
};

export const getAvailableTrains = async (source, destination) => {
  const safeSource = normalizeStationKey(source);
  const safeDestination = normalizeStationKey(destination);

  if (!safeSource || !safeDestination) {
    return [];
  }

  const trainsQuery = `
  MATCH (t:Train)-[:STOPS_AT]->(s1:Station),
        (t)-[:STOPS_AT]->(s2:Station)
  WHERE any(v IN [s1.name, s1.code, s1.stationCode, s1.station_name, s1.id]
            WHERE toLower(trim(toString(coalesce(v, "")))) = toLower($source))
    AND any(v IN [s2.name, s2.code, s2.stationCode, s2.station_name, s2.id]
            WHERE toLower(trim(toString(coalesce(v, "")))) = toLower($destination))
  RETURN DISTINCT
         coalesce(t.number, t.train_number, t.trainNo, t.no) AS number,
         coalesce(t.name, t.trainName, t.train_name, t.title, "") AS name,
      coalesce(t.type, t.trainType, "Express") AS type,
      id(t) AS graphId
  LIMIT 50
  `;

  try {
    const result = await runQuery(trainsQuery, {
      source: safeSource,
      destination: safeDestination,
    });
    const trains = result
      .map((row) => {
        const parsedNumber = toSafeNumber(row.number) ?? toSafeNumber(row.graphId);
        const rawName = String(row.name || "").trim();
        const nameNumberMatch = rawName.match(/\b(\d{4,6})\b/);
        const fallbackNumber = nameNumberMatch ? Number(nameNumberMatch[1]) : null;
        const effectiveNumber = parsedNumber || fallbackNumber;
        const name = rawName && rawName.toLowerCase() !== "unknown"
          ? rawName
          : (Number.isFinite(effectiveNumber) ? `Train ${effectiveNumber}` : "Train");

        return {
          number: effectiveNumber,
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
  const cacheKey = `${String(source).toLowerCase()}::${String(destination).toLowerCase()}`;
  const cached = routeFeatureCache.get(cacheKey);
  if (cached && Date.now() - cached.ts < ROUTE_CACHE_TTL_MS) {
    return cached.data;
  }

  const safeSource = normalizeStationKey(source);
  const safeDestination = normalizeStationKey(destination);

  if (!safeSource || !safeDestination) {
    return {
      distance: 560,
      stops: 8,
      time: 10.2,
      trainCount: 0
    };
  }

  const featureQuery = `
  MATCH (t:Train)-[:STOPS_AT]->(s1:Station),
        (t)-[:STOPS_AT]->(s2:Station)
  WHERE any(v IN [s1.name, s1.code, s1.stationCode, s1.station_name, s1.id]
            WHERE toLower(trim(toString(coalesce(v, "")))) = toLower($source))
    AND any(v IN [s2.name, s2.code, s2.stationCode, s2.station_name, s2.id]
            WHERE toLower(trim(toString(coalesce(v, "")))) = toLower($destination))
  WITH DISTINCT t
  OPTIONAL MATCH (t)-[:STOPS_AT]->(stop:Station)
  WITH t, count(stop) AS stopCount
  RETURN count(*) AS trainCount,
         avg(toFloat(stopCount)) AS avgStops,
         avg(CASE WHEN t.distance IS NULL THEN null ELSE toFloat(t.distance) END) AS avgDistance,
         avg(CASE WHEN t.travelTime IS NULL THEN null ELSE toFloat(t.travelTime) END) AS avgTravelTime
  `;

  const result = await runQuery(featureQuery, {
    source: safeSource,
    destination: safeDestination,
  });
  const route = result[0] || {};

  const trainCount = Number(route.trainCount) || 0;
  const stops = Math.max(2, Math.round(Number(route.avgStops) || trainCount || 8));
  const distance = Math.round(Number(route.avgDistance) || stops * 70);
  const time = Number((Number(route.avgTravelTime) || distance / 55).toFixed(1));

  const routeFeatures = {
    distance,
    stops,
    time,
    trainCount
  };

  routeFeatureCache.set(cacheKey, { ts: Date.now(), data: routeFeatures });
  return routeFeatures;
};