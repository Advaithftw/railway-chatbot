import axios from "axios";

export const getPrediction = async (features) => {
  try {
    const res = await axios.post("http://127.0.0.1:8000/predict", features);
    return res.data;
  } catch (err) {
    console.error("Prediction API error:", err.message);
    throw new Error("Prediction failed");
  }
};

export const getModelMetrics = async () => {
  try {
    const res = await axios.get("http://127.0.0.1:8000/metrics");
    return res.data;
  } catch (err) {
    console.error("Prediction metrics API error:", err.message);
    throw new Error("Failed to fetch model metrics");
  }
};