import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { ChatOpenAI } from "@langchain/openai";
import { MessagesAnnotation, StateGraph } from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";

// 1. First, let's create a simple database //rag
const weatherDatabase = {
  "New York": {
    temperature: 75,
    condition: "sunny",
    lastUpdated: "2024-03-20 10:00 AM"
  },
  "London": {
    temperature: 60,
    condition: "rainy",
    lastUpdated: "2024-03-20 3:00 PM"
  }
};

// 2. Create a tool to access this "real" data
const getWeather = tool(
  async ({ city }) => {
    // This could be a real API call in production
    const weatherData = weatherDatabase[city];
    if (!weatherData) return "City not found";
    return JSON.stringify(weatherData);
  },
  {
    name: "getWeather",
    description: "Get real-time weather for a city",
    schema: z.object({
      city: z.string().describe("name of the city"),
    }),
  }
);

// 3. Create a tool to log user preferences
const userPreferences = new Map();

const savePreference = tool(
  async ({ userId, preference }) => {
    userPreferences.set(userId, preference);
    return `Saved preference: ${preference} for user ${userId}`;
  },
  {
    name: "savePreference",
    description: "Save user weather preferences",
    schema: z.object({
      userId: z.string(),
      preference: z.string(),
    }),
  }
);

// 4. Set up the LLM and tools
const llm = new ChatOpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  modelName: "gpt-4",
});

const tools = [getWeather, savePreference];
const llmWithTools = llm.bindTools(tools);

// 5. Create the main function
async function llmCall(state) {
  const result = await llmWithTools.invoke([
    {
      role: "system",
      content: `You are a weather assistant. Ue the getWeather tool to fetch real wesather data.
                If users mention preferences, save them using savePreference.`
    },
    ...state.messages
  ]);

  return {
    messages: [result]
  };
}

// 6. Set up the workflow (same as before)
const toolNode = new ToolNode(tools);

function shouldContinue(state) {
  const lastMessage = state.messages.at(-1);
  return lastMessage?.tool_calls?.length ? "Action" : "__end__";
}

const weatherAssistant = new StateGraph(MessagesAnnotation)
  .addNode("llmCall", llmCall)
  .addNode("tools", toolNode)
  .addEdge("__start__", "llmCall")
  .addConditionalEdges(
    "llmCall",
    shouldContinue,
    {
      "Action": "tools",
      "__end__": "__end__",
    }
  )
  .addEdge("tools", "llmCall")
  .compile();

// 7. Example usage
async function demonstrateAssistant() {
  // First interaction
  console.log("=== First Interaction ===");
  const result1 = await weatherAssistant.invoke({
    messages: [{
      role: "user",
      content: "What's the weather in New York? I prefer warm weather."
    }]
  });
  console.log(result1.messages);

  // Second interaction
  console.log("\n=== Second Interaction ===");
  const result2 = await weatherAssistant.invoke({
    messages: [{
      role: "user",
      content: "What's the weather in London?"
    }]
  });
  console.log(result2.messages);
}

demonstrateAssistant(); 