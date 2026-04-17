import express from "express";
import { chatHandler, kgEvaluationHandler, mlEvaluationHandler } from "../controllers/chatController.js";

const router = express.Router();

router.post("/chat", chatHandler);
router.get("/evaluate/ml", mlEvaluationHandler);
router.get("/evaluate/kg", kgEvaluationHandler);

export default router;