import axios from "axios";

const OLLAMA_URL = process.env.OLLAMA_URL || "http://localhost:11434/api/generate";

const buildPrompt = (query) => `
Extract structured information from the query.

Return JSON ONLY.

Fields:
- waitlist (number)
- class (Sleeper, 3AC, 2AC)
- source (city)
- destination (city)
- trainNumber (number or null if not mentioned)

Query: "${query}"
`;

const parseJsonFromText = (text) => {
  const cleaned = String(text || "")
    .replace(/```json/gi, "")
    .replace(/```/g, "")
    .trim();

  try {
    return JSON.parse(cleaned);
  } catch {
    const objectMatch = cleaned.match(/\{[\s\S]*\}/);
    if (objectMatch) {
      return JSON.parse(objectMatch[0]);
    }
  }

  throw new Error("Failed to parse entity JSON from model output");
};

const normalizeEntities = (entities = {}) => {
  const normalizedClass = String(entities.class || "3AC").trim();
  const normalizeCity = (city = "") => {
    const trimmed = String(city).trim();
    if (trimmed.toLowerCase() === "banglore") return "Bangalore";
    return trimmed;
  };

  return {
    waitlist: Number(entities.waitlist) || 0,
    class: normalizedClass || "3AC",
    source: normalizeCity(entities.source || "") || "Unknown",
    destination: normalizeCity(entities.destination || "") || "Unknown",
    trainNumber: Number(entities.trainNumber) || null
  };
};

export const extractEntitiesRuleBased = (query) => {
  const text = String(query || "").toLowerCase();
  const cleanStation = (value = "") => {
    const normalized = String(value)
      .replace(/[^a-z\s]/gi, " ")
      .replace(/\s+/g, " ")
      .trim();

    if (!normalized) return "Unknown";

    // Remove trailing intent words from conversational queries.
    const withoutTrailingIntent = normalized.replace(
      /\b(?:will|can|could|should|get|be|confirm(?:ed)?|status|ticket|chance|probability)\b[\s\S]*$/i,
      ""
    ).trim();

    return withoutTrailingIntent || normalized;
  };
  const wlMatch = text.match(/\bwl\s*(\d+)\b|\bwaitlist\s*(\d+)\b|\b(\d+)\s*wl\b/i);
  const waitlist = Number(wlMatch?.[1] || wlMatch?.[2] || wlMatch?.[3]) || 0;

  const classMatch = text.match(/\b(1\s*ac|2\s*ac|3\s*ac|a1|a2|a3|sleeper|sl)\b/i);
  let travelClass = "3AC";
  if (classMatch) {
    const cls = classMatch[1].replace(/\s+/g, "").toLowerCase();
    if (cls === "1ac" || cls === "a1") travelClass = "1AC";
    else if (cls === "2ac" || cls === "a2") travelClass = "2AC";
    else if (cls === "3ac" || cls === "a3") travelClass = "3AC";
    else travelClass = "Sleeper";
  }

  const routeMatch = text.match(
    /from\s+([a-z\s]+?)\s+to\s+([a-z\s]+?)(?=\s+(?:will|can|could|should|get|be|confirm(?:ed)?|status|ticket|chance|probability)\b|\?|\.|,|$)/i
  );
  const source = cleanStation(routeMatch?.[1] || "Unknown").replace(/^banglore$/i, "Bangalore");
  const destination = cleanStation(routeMatch?.[2] || "Unknown").replace(/^banglore$/i, "Bangalore");

  const trainMatch = text.match(/train\s*(\d+)\b|\b(\d{5})\b/i);
  const trainNumber = Number(trainMatch?.[1] || trainMatch?.[2]) || null;

  return {
    waitlist,
    class: travelClass,
    source,
    destination,
    trainNumber
  };
};

const tryOllamaExtraction = async (query, model, options = undefined) => {
  const res = await axios.post(OLLAMA_URL, {
    model,
    prompt: buildPrompt(query),
    stream: false,
    options
  });

  return normalizeEntities(parseJsonFromText(res.data?.response));
};

const isPlausibleExtraction = (entities = {}) => {
  const source = String(entities.source || "").trim().toLowerCase();
  const destination = String(entities.destination || "").trim().toLowerCase();
  const travelClass = String(entities.class || "").trim().toLowerCase();
  const validClasses = new Set(["2ac", "3ac", "sleeper"]);

  // Reject generic placeholders and malformed classes from noisy model output.
  if (!source || !destination) return false;
  if (source === "unknown" || destination === "unknown") return false;
  if (source === "city" || destination === "city") return false;
  if (!validClasses.has(travelClass)) return false;

  return true;
};

export const extractEntities = async (query) => {
  const configuredModels = (process.env.OLLAMA_ENTITY_MODELS || "llama3,mistral,phi3,tinyllama")
    .split(",")
    .map((m) => m.trim())
    .filter(Boolean);

  for (const model of configuredModels) {
    try {
      const entities = await tryOllamaExtraction(query, model);
      if (isPlausibleExtraction(entities)) {
        return entities;
      }
      console.warn(`Entity extraction produced implausible output with model ${model}:`, entities);
    } catch (err) {
      const message = err?.response?.data?.error || err.message;
      console.warn(`Entity extraction failed with model ${model}:`, message);
    }
  }

  try {
    const cpuEntities = await tryOllamaExtraction(query, configuredModels[0] || "llama3", { num_gpu: 0 });
    if (isPlausibleExtraction(cpuEntities)) {
      return cpuEntities;
    }
    console.warn("Entity extraction CPU retry produced implausible output:", cpuEntities);
  } catch (err) {
    const message = err?.response?.data?.error || err.message;
    console.warn("Entity extraction CPU retry failed:", message);
  }

  return extractEntitiesRuleBased(query);
};

export const extractEntitiesFast = async (query) => {
  // Fast path for latency-sensitive prediction queries.
  return extractEntitiesRuleBased(query);
};