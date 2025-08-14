import { StateGraph, START, END } from "@langchain/langgraph";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { z } from "zod";
import dotenv from 'dotenv';
import { callLLMAPI } from './llm.js';
import { getCachedDataAnalysis } from '../data/data_exploration.js';
import { codeGenSystem, codeReviewSystem, domainHPSystem } from './agent_prompts.js';

dotenv.config();

// Modelle (kannst du auf Ollama o.ä. wechseln)
function createChatModel(modelName = process.env.LLM_CODE_MODEL || 'gemini-2.5-flash-lite') {
  return new ChatGoogleGenerativeAI({
    model: modelName,
    apiKey: process.env.GEMINI_API_KEY,
    temperature: 0.2
  });
}

export async function runMultiAgentPipeline(project, modelName = process.env.LLM_CODE_MODEL || 'gemini-2.5-flash-lite') {
  // const llm = createChatModel(modelName); // aktuell nicht direkt übergeben, Modellname wird an callLLMAPI gereicht

  // Nodes
  const domainHyperparamsNode = async (state) => {
    console.log('[Multi-Agent] Starte Agent: Domain-Hyperparameter');
    const { project, dataAnalysis } = state || {};
    const prompt = domainHPSystem(dataAnalysis);
    const suggestionStr = await callLLMAPI(prompt, null, modelName);
    let suggestion = {};
    try { suggestion = JSON.parse(typeof suggestionStr === 'string' ? suggestionStr : suggestionStr?.result || '{}'); } catch {}
    console.log('[Multi-Agent] Fertig: Domain-Hyperparameter');
    return { hyperparamsSuggestion: suggestion };
  };

  const codeGenNode = async (state) => {
    console.log('[Multi-Agent] Starte Agent: Code-Generator');
    const { project, dataAnalysis, hyperparamsSuggestion } = state || {};
    const prompt = codeGenSystem(project || {}, dataAnalysis || {}, hyperparamsSuggestion || {});
    const code = await callLLMAPI(prompt, null, modelName);
    const text = typeof code === 'string' ? code : (code?.result || '');
    const block = text.match(/```(?:python)?\n([\s\S]*?)```/i);
    const pythonCode = (block ? block[1] : text).trim();
    // Lösche die erste Zeile, die mit ```python und #!/usr/bin/env python3 beginnt
    if (pythonCode.startsWith('```python\n#!/usr/bin/env python3\n')) {
      pythonCode = pythonCode.replace('```python\n#!/usr/bin/env python3\n', '');
    }
    // Lösche die letzte Zeile, die mit ``` endet
    if (pythonCode.endsWith('\n```\n')) {
      pythonCode = pythonCode.replace('\n```\n', '\n');
    }

    console.log('[Multi-Agent] Fertig: Code-Generator');
    return { pythonCode };
  };

  const codeReviewNode = async (state) => {
    console.log('[Multi-Agent] Starte Agent: Code-Review');
    const prompt = codeReviewSystem(state?.pythonCode || '');
    const reviewed = await callLLMAPI(prompt, null, modelName);
    const text = typeof reviewed === 'string' ? reviewed : (reviewed?.result || '');
    const block = text.match(/```(?:python)?\n([\s\S]*?)```/i);
    let reviewCode = (block ? block[1] : text).trim();
    // Lösche die erste Zeile, die mit ```python und #!/usr/bin/env python3 beginnt
    if (reviewCode.startsWith('```python\n#!/usr/bin/env python3\n')) {
      reviewCode = reviewCode.replace('```python\n#!/usr/bin/env python3\n', '');
    }
    // Lösche die letzte Zeile, die mit ``` endet
    if (reviewCode.endsWith('\n```\n')) {
      reviewCode = reviewCode.replace('\n```\n', '\n');
    }
    console.log('[Multi-Agent] Fertig: Code-Review');
    return { reviewCode };
  };

  // Graph aufbauen
  // State-Schema (Zod) für LangGraph
  const GraphState = z.object({
    project: z.any().optional(),
    dataAnalysis: z.any().optional(),
    hyperparamsSuggestion: z.any().optional(),
    pythonCode: z.string().optional(),
    reviewCode: z.string().optional(),
  });

  const graph = new StateGraph(GraphState)
    .addNode("domainHP", domainHyperparamsNode)
    .addNode("codeGen", codeGenNode)
    .addNode("review", codeReviewNode)
    .addEdge(START, "domainHP")
    .addEdge("domainHP", "codeGen")
    .addEdge("codeGen", "review")
    .addEdge("review", END);

  const app = graph.compile();

  console.log('Multi-Agent Orchestrator started');

  // Datenanalyse laden
  let dataAnalysis = null;
  try {
    dataAnalysis = await getCachedDataAnalysis(project.csvFilePath);
  } catch {}

  const initialState = { project, dataAnalysis };
  const finalState = await app.invoke(initialState);
  return finalState?.reviewCode || finalState?.pythonCode;
}


