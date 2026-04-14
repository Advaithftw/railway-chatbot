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

const extractEntitiesRuleBased = (query) => {
  const text = String(query || "").toLowerCase();
  const wlMatch = text.match(/\bwl\s*(\d+)\b|\bwaitlist\s*(\d+)\b|\b(\d+)\s*wl\b/i);
  const waitlist = Number(wlMatch?.[1] || wlMatch?.[2] || wlMatch?.[3]) || 0;

  const classMatch = text.match(/\b(2\s*ac|3\s*ac|sleeper|sl)\b/i);
  let travelClass = "3AC";
  if (classMatch) {
    const cls = classMatch[1].replace(/\s+/g, "").toLowerCase();
    if (cls === "2ac") travelClass = "2AC";
    else if (cls === "3ac") travelClass = "3AC";
    else travelClass = "Sleeper";
  }

  const routeMatch = text.match(/from\s+([a-z\s]+?)\s+to\s+([a-z\s]+?)(?:\?|\.|,|$)/i);
  const source = (routeMatch?.[1]?.trim() || "Unknown").replace(/^banglore$/i, "Bangalore");
  const destination = (routeMatch?.[2]?.trim() || "Unknown").replace(/^banglore$/i, "Bangalore");

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

export const extractEntities = async (query) => {
  const configuredModels = (process.env.OLLAMA_ENTITY_MODELS || "llama3,mistral,phi3,tinyllama")
    .split(",")
    .map((m) => m.trim())
    .filter(Boolean);

  for (const model of configuredModels) {
    try {
      return await tryOllamaExtraction(query, model);
    } catch (err) {
      const message = err?.response?.data?.error || err.message;
      console.warn(`Entity extraction failed with model ${model}:`, message);
    }
  }

  try {
    return await tryOllamaExtraction(query, configuredModels[0] || "llama3", { num_gpu: 0 });
  } catch (err) {
    const message = err?.response?.data?.error || err.message;
    console.warn("Entity extraction CPU retry failed:", message);
  }

  return extractEntitiesRuleBased(query);
};