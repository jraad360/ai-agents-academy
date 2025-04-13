import { Module } from '../../../types';

export const componentsModule: Module = {
  id: 'components',
  title: 'Agent Components',
  description: 'Understanding the building blocks of AI agents',
  duration: '1 hour',
  level: 'beginner',
  content: {
    sections: [
      {
        title: 'Tools and Tool Usage',
        content: `Tools are functions that agents can use to interact with their environment or perform specific tasks. LangChain provides a rich ecosystem of tools and the ability to create custom ones.

Learn more about tools in the [official documentation](https://python.langchain.com/docs/modules/agents/tools/).

Tool Components:
1. Name - Unique identifier
2. Description - Explains tool functionality
3. Function - Implementation logic
4. Schema - Input/output specifications

Tool Categories:
• Built-in Tools
  - Calculators
  - Web browsers
  - Shell commands
  
• External Integrations
  - APIs
  - Databases
  - Search engines

• Custom Tools
  - Domain-specific functions
  - Business logic
  - Internal services

Best Practices:
• Clear descriptions for better tool selection
• Proper error handling
• Input validation
• Consistent output formats`,
        codeExamples: [{
          typescript: `import { DynamicTool, Tool } from "@langchain/core/tools";
import { z } from "zod";

// Create a structured tool with input/output schemas
const weatherTool = new Tool({
  name: "weather",
  description: "Get current weather for a location",
  schema: z.object({
    location: z.string().describe("The city or location to get weather for")
  }),
  async func({ location }) {
    try {
      // Implement weather API call
      return { temperature: 72, conditions: "sunny" };
    } catch (error) {
      throw new Error(\`Failed to get weather: \${error.message}\`);
    }
  }
});

// Create a dynamic tool
const databaseTool = new DynamicTool({
  name: "database",
  description: "Query the product database",
  func: async (query: string) => {
    try {
      // Implement database query
      return JSON.stringify({ results: [] });
    } catch (error) {
      throw new Error(\`Database query failed: \${error.message}\`);
    }
  }
});

// Use tools with proper error handling
try {
  const weather = await weatherTool.invoke({ location: "Paris" });
  console.log(weather);
} catch (error) {
  console.error("Tool execution failed:", error);
}`,
          python: `from langchain.tools import Tool, StructuredTool
from langchain.pydantic_v1 import BaseModel, Field
from typing import Optional

# Define input schema
class WeatherInput(BaseModel):
    location: str = Field(..., description="The city or location to get weather for")

# Create a structured tool with input schema
weather_tool = StructuredTool.from_function(
    name="weather",
    description="Get current weather for a location",
    func=lambda x: {"temperature": 72, "conditions": "sunny"},
    args_schema=WeatherInput
)

# Create a dynamic tool
database_tool = Tool.from_function(
    name="database",
    description="Query the product database",
    func=lambda query: {"results": []}
)

# Use tools with proper error handling
try:
    weather = weather_tool.invoke({"location": "Paris"})
    print(weather)
except Exception as e:
    print(f"Tool execution failed: {str(e)}")`,
          description: 'Creating and using tools with proper schemas and error handling'
        }]
      },
      {
        title: 'Memory Systems',
        content: `Memory systems in LangChain enable agents to maintain context and recall information from previous interactions. Learn more about memory types in the [documentation](https://python.langchain.com/docs/modules/memory/).

Memory Types:

1. Buffer Memory
   - Stores recent conversations
   - Simple, FIFO structure
   - Good for basic context

2. Vector Memory
   - Semantic search capabilities
   - Scales to large histories
   - Efficient retrieval

3. Entity Memory
   - Tracks specific entities
   - Maintains attributes
   - Relationship tracking

4. Summary Memory
   - Compressed history
   - Key points retention
   - Efficient for long conversations

Best Practices:
• Choose appropriate memory type
• Implement proper cleanup
• Handle memory limits
• Consider privacy implications`,
        codeExamples: [{
          typescript: `import { BufferMemory, VectorStoreRetrieverMemory } from "langchain/memory";
import { HNSWLib } from "@langchain/community/vectorstores/hnswlib";
import { OpenAIEmbeddings } from "@langchain/openai";
import { ChatOpenAI } from "@langchain/openai";
import { MessagesPlaceholder } from "@langchain/core/prompts";

// Create a chat model
const model = new ChatOpenAI({
  modelName: "gpt-4",
  temperature: 0
});

// Simple buffer memory
const bufferMemory = new BufferMemory({
  returnMessages: true,
  memoryKey: "chat_history",
  inputKey: "input",
  outputKey: "output"
});

// Vector store memory for semantic search
const vectorStore = await HNSWLib.fromTexts(
  ["previous conversation"],
  [{ id: 1 }],
  new OpenAIEmbeddings()
);

const vectorMemory = new VectorStoreRetrieverMemory({
  vectorStoreRetriever: vectorStore.asRetriever(),
  memoryKey: "semantic_history"
});

// Create a chain with memory
const chain = model.bind({
  memory: bufferMemory,
  prompt: new MessagesPlaceholder("chat_history")
});

// Use the chain
const result = await chain.invoke({
  input: "What did we discuss earlier?"
});`,
          python: `from langchain_openai import ChatOpenAI
from langchain.memory import BufferMemory, VectorStoreRetrieverMemory
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import MessagesPlaceholder
from langchain.chains import LLMChain

# Create a chat model
model = ChatOpenAI(
    model_name="gpt-4",
    temperature=0
)

# Simple buffer memory
buffer_memory = BufferMemory(
    return_messages=True,
    memory_key="chat_history",
    input_key="input",
    output_key="output"
)

# Vector store memory for semantic search
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_texts(
    ["previous conversation"],
    embedding=embeddings
)

vector_memory = VectorStoreRetrieverMemory(
    retriever=vector_store.as_retriever(),
    memory_key="semantic_history"
)

# Create a chain with memory
chain = LLMChain(
    llm=model,
    memory=buffer_memory,
    prompt=MessagesPlaceholder(variable_name="chat_history")
)

# Use the chain
result = chain.invoke({
    "input": "What did we discuss earlier?"
})`,
          description: 'Implementing different types of memory systems'
        }]
      }
    ]
  }
};