import axios from "axios";

export const generateCypher = async (userQuery) => {
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

  const response = await axios.post("http://localhost:11434/api/generate", {
    model: "llama3",
    prompt: prompt,
    stream: false
  });

  return response.data.response.trim();
};

export const generateAnswer = async (question, data) => {
  if (!data || data.length === 0) return "No results found.";

  const trains = data
    .map(d => d.train)   // clean key now
    .filter(Boolean);

  return `Found ${trains.length} trains:\n- ` + trains.join("\n- ");
};