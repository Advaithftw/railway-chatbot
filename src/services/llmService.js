import axios from "axios";

const OLLAMA_URL = process.env.OLLAMA_URL || "http://localhost:11434/api/generate";

const escapeForCypher = (value = "") =>
  String(value).replace(/\\/g, "\\\\").replace(/"/g, '\\"').trim();

const tryExtractCypher = (text = "") => {
  const cleaned = String(text)
    .replace(/```cypher/gi, "")
    .replace(/```/g, "")
    .trim();

  const lines = cleaned
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);

  const startIndex = lines.findIndex((line) =>
    /^(MATCH|WITH|UNWIND|CALL|OPTIONAL MATCH)\b/i.test(line)
  );

  if (startIndex === -1) return null;

  const cypherLines = [];
  for (let i = startIndex; i < lines.length; i += 1) {
    const line = lines[i];
    if (/^(explanation|note|here('|’)s|you can|this query)/i.test(line)) break;
    cypherLines.push(line);
  }

  const candidate = cypherLines.join("\n").trim();
  if (!candidate) return null;
  if (!/^(MATCH|WITH|UNWIND|CALL|OPTIONAL MATCH)\b/i.test(candidate)) return null;
  if (!/\bRETURN\b/i.test(candidate)) return null;
  if (candidate.includes("<city>") || candidate.includes("<station>")) return null;
  return candidate;
};

const buildRuleBasedCypher = (query = "") => {
  const text = String(query).toLowerCase().trim();
  const routeMatch = text.match(/from\s+([a-z\s]+?)\s+to\s+([a-z\s]+?)(?:\?|\.|,|$)/i);
  const fromMatch = text.match(/from\s+([a-z\s]+?)(?:\?|\.|,|$)/i);
  const toMatch = text.match(/to\s+([a-z\s]+?)(?:\?|\.|,|$)/i);

  if (routeMatch) {
    const source = escapeForCypher(routeMatch[1].toLowerCase());
    const destination = escapeForCypher(routeMatch[2].toLowerCase());
    return `
MATCH (t:Train)-[:STOPS_AT]->(s1:Station),
      (t)-[:STOPS_AT]->(s2:Station)
WHERE toLower(s1.name) CONTAINS "${source}"
  AND toLower(s2.name) CONTAINS "${destination}"
RETURN DISTINCT t.name AS train, t.number AS number, t.type AS type
LIMIT 50
`.trim();
  }

  if (fromMatch || toMatch) {
    const station = escapeForCypher((fromMatch?.[1] || toMatch?.[1] || "").toLowerCase());
    return `
MATCH (t:Train)-[:STOPS_AT]->(s:Station)
WHERE toLower(s.name) CONTAINS "${station}"
RETURN DISTINCT t.name AS train, t.number AS number, t.type AS type
LIMIT 50
`.trim();
  }

  return `
MATCH (t:Train)
RETURN DISTINCT t.name AS train, t.number AS number, t.type AS type
LIMIT 50
`.trim();
};

export const generateCypher = async (userQuery) => {
  const normalizedQuery = String(userQuery || "").toLowerCase();
  // Prefer deterministic route extraction for common railway questions.
  if (/\btrains?\b/.test(normalizedQuery) && /\b(from|to)\b/.test(normalizedQuery)) {
    return buildRuleBasedCypher(userQuery);
  }

  const prompt = `
You are a Neo4j Cypher expert.

Schema:
- Train(name, number)
- Station(name, code)
- (Train)-[:STOPS_AT]->(Station)

Rules:
1. "Chennai", "Delhi", etc are ALWAYS Station names
2. NEVER match Train.name with city names
3. ALWAYS match stations using:
   toLower(s.name) CONTAINS "<city>"
4. ALWAYS use this pattern:

MATCH (t:Train)-[:STOPS_AT]->(s:Station)
WHERE toLower(s.name) CONTAINS "<city>"
RETURN DISTINCT t.name

5. Return ONLY Cypher query
6. No explanation, no markdown, no extra text

Examples:

User: trains from Chennai
Cypher:
MATCH (t:Train)-[:STOPS_AT]->(s:Station)
WHERE toLower(s.name) CONTAINS "chennai"
RETURN DISTINCT t.name

User: trains from Delhi
Cypher:
MATCH (t:Train)-[:STOPS_AT]->(s:Station)
WHERE toLower(s.name) CONTAINS "delhi"
RETURN DISTINCT t.name

---

User: ${userQuery}
Cypher:
`;

  const configuredModels = (process.env.OLLAMA_CYPHER_MODELS || process.env.OLLAMA_ENTITY_MODELS || "tinyllama")
    .split(",")
    .map((m) => m.trim())
    .filter(Boolean);

  for (const model of configuredModels) {
    try {
      const response = await axios.post(OLLAMA_URL, {
        model,
        prompt,
        stream: false
      });

      const cypher = tryExtractCypher(response.data?.response);
      if (cypher) {
        return cypher;
      }
    } catch (err) {
      const message = err?.response?.data?.error || err.message;
      console.warn(`Cypher generation failed with model ${model}:`, message);
    }
  }

  return buildRuleBasedCypher(userQuery);
};

export const generateAnswer = async (question, data) => {
  if (!data || data.length === 0) return "No results found.";

  const trains = data
    .map(d => d.train)   // clean key now
    .filter(Boolean);

  return `Found ${trains.length} trains:\n- ` + trains.join("\n- ");
};