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