import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { ChatOpenAI } from "@langchain/openai";
import { MessagesAnnotation, StateGraph } from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { config } from "dotenv";

config();

// Initialize our LLM
const llm = new ChatOpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  modelName: "gpt-4",
});

// Tool to search for activities in a city
const searchActivities = tool(
  async ({ city, category }) => {
    // In a real implementation, this would call an API
    const mockDatabase = {
      "New York": {
        "museums": ["Metropolitan Museum of Art", "MoMA", "Natural History Museum"],
        "parks": ["Central Park", "High Line", "Bryant Park"],
        "entertainment": ["Broadway Shows", "Times Square", "Madison Square Garden"]
      },
      "London": {
        "museums": ["British Museum", "Tate Modern", "Natural History Museum"],
        "parks": ["Hyde Park", "Regent's Park", "Greenwich Park"],
        "entertainment": ["West End Shows", "O2 Arena", "Shakespeare's Globe"]
      }
    };

    return JSON.stringify(mockDatabase[city]?.[category] || "City or category not found");
  },
  {
    name: "searchActivities",
    description: "Search for activities in a specific city by category",
    schema: z.object({
      city: z.string().describe("name of the city"),
      category: z.string().describe("category of activity (museums, parks, entertainment)"),
    }),
  }
);

// Tool to get restaurant recommendations
const getRestaurants = tool(
  async ({ city, cuisine, priceRange }) => {
    // Mock database - in reality, this would be an API call
    const mockRestaurants = {
      "New York": {
        "italian": {
          "budget": ["Pizza Palace", "Mama's Kitchen"],
          "luxury": ["Del Posto", "Il Mulino"]
        },
        "japanese": {
          "budget": ["Sushi Express", "Ramen House"],
          "luxury": ["Nobu", "Morimoto"]
        }
      }
    };

    return JSON.stringify(
      mockRestaurants[city]?.[cuisine]?.[priceRange] ||
      "No restaurants found matching criteria"
    );
  },
  {
    name: "getRestaurants",
    description: "Get restaurant recommendations based on city, cuisine, and price range",
    schema: z.object({
      city: z.string().describe("name of the city"),
      cuisine: z.string().describe("type of cuisine"),
      priceRange: z.string().describe("budget or luxury"),
    }),
  }
);

// Tool to create an itinerary
const createItinerary = tool(
  async ({ activities, restaurants, days }) => {
    // Simple itinerary generator
    const itinerary = {};
    const allActivities = JSON.parse(activities);
    const allRestaurants = JSON.parse(restaurants);

    for (let i = 1; i <= days; i++) {
      itinerary[`Day ${i}`] = {
        "Morning": allActivities[i % allActivities.length],
        "Lunch": allRestaurants[0],
        "Afternoon": allActivities[(i + 1) % allActivities.length],
        "Dinner": allRestaurants[1]
      };
    }

    return JSON.stringify(itinerary, null, 2);
  },
  {
    name: "createItinerary",
    description: "Create a daily itinerary based on selected activities and restaurants",
    schema: z.object({
      activities: z.string().describe("JSON string of activities"),
      restaurants: z.string().describe("JSON string of restaurants"),
      days: z.number().describe("number of days"),
    }),
  }
);

// Collect all tools
const tools = [searchActivities, getRestaurants, createItinerary];
const llmWithTools = llm.bindTools(tools);

// Main LLM function
async function llmCall(state) {
  const result = await llmWithTools.invoke([
    {
      role: "system",
      content: `You are a helpful travel assistant that helps users plan their trips.
                You can search for activities, find restaurants, and create itineraries.
                Always try to understand the user's preferences and make personalized recommendations.`
    },
    ...state.messages
  ]);

  return {
    messages: [result]
  };
}

// Create tool node
const toolNode = new ToolNode(tools);

// Decision function
function shouldContinue(state) {
  const messages = state.messages;
  const lastMessage = messages.at(-1);
  return lastMessage?.tool_calls?.length ? "Action" : "__end__";
}

// Build workflow
const travelAssistant = new StateGraph(MessagesAnnotation)
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

// Example usage
const messages = [{
  role: "user",
  content: "I'm planning a 2-day trip to New York. I love museums and Italian food, and I prefer luxury dining. Can you help me plan my trip?"
}];

// Run the assistant
async function runAssistant() {
  const result = await travelAssistant.invoke({ messages });
  console.log(JSON.stringify(result.messages, null, 2));
}

runAssistant(); 