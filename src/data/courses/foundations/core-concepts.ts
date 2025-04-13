import { Module } from '../../../types';

export const coreConceptsModule: Module = {
  id: 'core-concepts',
  title: 'LangChain Core Concepts',
  description: 'Understanding the fundamental building blocks of LangChain',
  duration: '1.5 hours',
  level: 'beginner',
  content: {
    sections: [
      {
        title: 'Language Models',
        content: `Language models are the foundation of AI agents, providing the reasoning and generation capabilities. LangChain supports two main types of language models:

1. Chat Models
   - Process sequences of messages as input
   - Return a message as output
   - Support for various roles (system, user, assistant)
   - Examples: GPT-4, Claude, Llama, Mistral

2. Text LLMs (Legacy)
   - Take a string as input
   - Return a string as output
   - Simpler interface but less context control
   - Being phased out in favor of Chat Models

Key considerations when working with language models:

• Context Windows
  - Maximum size of input a model can process
  - Varies by model (e.g., 8K, 16K, 32K, 128K tokens)
  - Requires strategies for handling long inputs

• Temperature and Sampling
  - Temperature controls randomness (0.0 = deterministic, 1.0 = creative)
  - Top-p and top-k sampling for controlling output diversity
  - Choosing appropriate settings based on task requirements

• Model Selection
  - Balancing capability vs. cost
  - Open vs. closed source considerations
  - Specialized models for specific tasks

Learn more about language models in the [LangChain documentation](https://python.langchain.com/docs/concepts/chat_models/).`,
        codeExamples: [{
          typescript: `import { ChatOpenAI } from "@langchain/openai";
import { ChatAnthropic } from "@langchain/anthropic";
import { ChatOllama } from "@langchain/ollama";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";

// Initialize different chat models
const openAIModel = new ChatOpenAI({
  modelName: "gpt-4o",
  temperature: 0,
  maxTokens: 500
});

const anthropicModel = new ChatAnthropic({
  modelName: "claude-3-5-sonnet-20240620",
  temperature: 0.2
});

// Local model via Ollama
const localModel = new ChatOllama({
  model: "llama3",
  temperature: 0
});

// Create messages
const messages = [
  new SystemMessage("You are a helpful AI assistant that provides concise answers."),
  new HumanMessage("What is the capital of France?")
];

// Invoke the model
const response = await openAIModel.invoke(messages);
console.log(response.content);

// Stream the response
const stream = await anthropicModel.stream(messages);
for await (const chunk of stream) {
  process.stdout.write(chunk.content);
}`,
          python: `from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

# Initialize different chat models
openai_model = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=500
)

anthropic_model = ChatAnthropic(
    model="claude-3-5-sonnet-20240620",
    temperature=0.2
)

# Local model via Ollama
local_model = ChatOllama(
    model="llama3",
    temperature=0
)

# Create messages
messages = [
    SystemMessage(content="You are a helpful AI assistant that provides concise answers."),
    HumanMessage(content="What is the capital of France?")
]

# Invoke the model
response = openai_model.invoke(messages)
print(response.content)

# Stream the response
stream = anthropic_model.stream(messages)
for chunk in stream:
    print(chunk.content, end="")`,
          description: 'Working with different chat models and streaming responses'
        }]
      },
      {
        title: 'LangChain Expression Language (LCEL)',
        content: `LangChain Expression Language (LCEL) is a declarative way to compose LangChain components into chains. It provides a simple, intuitive syntax for building complex workflows while optimizing runtime execution.

Key benefits of LCEL:

• Optimized parallel execution
• Guaranteed async support
• Simplified streaming
• Seamless LangSmith tracing
• Standard API across components
• Deployable with LangServe

LCEL is built on the Runnable interface, which provides a standard way to interact with all LangChain components. The two main composition primitives are:

1. RunnableSequence
   - Chain components sequentially
   - Output of one component becomes input to the next
   - Created using the pipe (|) operator or .pipe() method

2. RunnableParallel
   - Run multiple components with the same input
   - Results combined into a dictionary
   - Created using a dictionary of runnables

When to use LCEL:
- For simple chains (prompt + model + parser)
- For basic retrieval setups
- When you need optimized execution

When to use LangGraph instead:
- For complex workflows with branching
- For multi-agent systems
- For stateful applications with cycles

Learn more about LCEL in the [LangChain documentation](https://python.langchain.com/docs/concepts/lcel/).`,
        codeExamples: [{
          typescript: `import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnableSequence, RunnableParallel } from "@langchain/core/runnables";

// Create a chat model
const model = new ChatOpenAI({
  temperature: 0
});

// Create a prompt template
const promptTemplate = ChatPromptTemplate.fromMessages([
  ["system", "You are a helpful assistant."],
  ["human", "Tell me about {topic} in {style} style."]
]);

// Create an output parser
const outputParser = new StringOutputParser();

// Sequential chain using the pipe operator
const sequentialChain = promptTemplate
  .pipe(model)
  .pipe(outputParser);

// Equivalent using RunnableSequence
const explicitSequentialChain = RunnableSequence.from([
  promptTemplate,
  model,
  outputParser
]);

// Invoke the sequential chain
const result = await sequentialChain.invoke({
  topic: "artificial intelligence",
  style: "concise"
});

// Parallel chain using RunnableParallel
const parallelChain = RunnableParallel.from({
  summary: promptTemplate
    .pipe(model)
    .pipe(outputParser),
  keywords: ChatPromptTemplate.fromMessages([
    ["system", "Extract 5 keywords from the topic. Return as comma-separated list."],
    ["human", "{topic}"]
  ])
    .pipe(model)
    .pipe(outputParser)
});

// Invoke the parallel chain
const parallelResult = await parallelChain.invoke({
  topic: "artificial intelligence",
  style: "concise"
});

// Access results
console.log("Summary:", parallelResult.summary);
console.log("Keywords:", parallelResult.keywords);`,
          python: `from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel

# Create a chat model
model = ChatOpenAI(temperature=0)

# Create a prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Tell me about {topic} in {style} style.")
])

# Create an output parser
output_parser = StrOutputParser()

# Sequential chain using the pipe operator
sequential_chain = prompt_template | model | output_parser

# Equivalent using RunnableSequence
explicit_sequential_chain = RunnableSequence.from_components([
    prompt_template,
    model,
    output_parser
])

# Invoke the sequential chain
result = sequential_chain.invoke({
    "topic": "artificial intelligence",
    "style": "concise"
})

# Parallel chain using RunnableParallel
parallel_chain = RunnableParallel(
    summary=prompt_template | model | output_parser,
    keywords=ChatPromptTemplate.from_messages([
        ("system", "Extract 5 keywords from the topic. Return as comma-separated list."),
        ("human", "{topic}")
    ]) | model | output_parser
)

# Invoke the parallel chain
parallel_result = parallel_chain.invoke({
    "topic": "artificial intelligence",
    "style": "concise"
})

# Access results
print("Summary:", parallel_result["summary"])
print("Keywords:", parallel_result["keywords"])`,
          description: 'Building chains with LCEL using sequential and parallel composition'
        }]
      },
      {
        title: 'Prompt Engineering',
        content: `Prompt engineering is the practice of designing effective prompts to guide language model behavior. Well-crafted prompts can significantly improve the quality, relevance, and accuracy of model outputs.

Key prompt engineering techniques:

1. Prompt Templates
   - Reusable structures with variable placeholders
   - Separate static and dynamic content
   - Support for different message types (system, human, assistant)

2. Few-Shot Learning
   - Including examples in the prompt
   - Demonstrates desired input-output patterns
   - Helps models understand the expected format and style

3. Chain-of-Thought Prompting
   - Encourages step-by-step reasoning
   - Improves performance on complex tasks
   - Reduces logical errors

4. Structured Outputs
   - Guiding models to produce specific formats
   - Using output parsers to validate and extract data
   - Handling parsing errors gracefully

Best practices:
• Be clear and specific about the task
• Provide context and constraints
• Use system messages for persistent instructions
• Balance brevity and detail
• Test and iterate on prompts

Learn more about prompt engineering in the [LangChain documentation](https://python.langchain.com/docs/concepts/prompt_templates/).`,
        codeExamples: [{
          typescript: `import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage, AIMessage } from "@langchain/core/messages";

// Basic prompt template
const basicPrompt = ChatPromptTemplate.fromMessages([
  ["system", "You are a helpful assistant that specializes in {domain}."],
  ["human", "{input}"]
]);

// Few-shot prompt template with examples
const fewShotPrompt = ChatPromptTemplate.fromMessages([
  ["system", "You are a helpful assistant that translates English to French."],
  ["human", "Hello, how are you?"],
  ["ai", "Bonjour, comment allez-vous?"],
  ["human", "I love artificial intelligence."],
  ["ai", "J'adore l'intelligence artificielle."],
  ["human", "{input}"]
]);

// Chain-of-thought prompt
const cotPrompt = ChatPromptTemplate.fromMessages([
  ["system", "You are a math tutor. Solve problems step-by-step, showing your work."],
  ["human", "{problem}"]
]);

// Prompt with conversation history
const conversationPrompt = ChatPromptTemplate.fromMessages([
  ["system", "You are a helpful assistant."],
  new MessagesPlaceholder("history"),
  ["human", "{input}"]
]);

// Initialize the model
const model = new ChatOpenAI();

// Use the basic prompt
const basicResult = await model.invoke(
  await basicPrompt.formatMessages({
    domain: "computer science",
    input: "Explain what a binary tree is."
  })
);

// Use the few-shot prompt
const fewShotResult = await model.invoke(
  await fewShotPrompt.formatMessages({
    input: "The weather is nice today."
  })
);

// Use the chain-of-thought prompt
const cotResult = await model.invoke(
  await cotPrompt.formatMessages({
    problem: "If a train travels 120 miles in 2 hours, what is its average speed?"
  })
);

// Use the conversation prompt with history
const history = [
  new HumanMessage("What is the capital of France?"),
  new AIMessage("The capital of France is Paris.")
];

const conversationResult = await model.invoke(
  await conversationPrompt.formatMessages({
    history,
    input: "What about Germany?"
  })
);`,
          python: `from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

# Basic prompt template
basic_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that specializes in {domain}."),
    ("human", "{input}")
])

# Few-shot prompt template with examples
few_shot_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that translates English to French."),
    ("human", "Hello, how are you?"),
    ("ai", "Bonjour, comment allez-vous?"),
    ("human", "I love artificial intelligence."),
    ("ai", "J'adore l'intelligence artificielle."),
    ("human", "{input}")
])

# Chain-of-thought prompt
cot_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a math tutor. Solve problems step-by-step, showing your work."),
    ("human", "{problem}")
])

# Prompt with conversation history
conversation_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Initialize the model
model = ChatOpenAI()

# Use the basic prompt
basic_result = model.invoke(
    basic_prompt.format_messages(
        domain="computer science",
        input="Explain what a binary tree is."
    )
)

# Use the few-shot prompt
few_shot_result = model.invoke(
    few_shot_prompt.format_messages(
        input="The weather is nice today."
    )
)

# Use the chain-of-thought prompt
cot_result = model.invoke(
    cot_prompt.format_messages(
        problem="If a train travels 120 miles in 2 hours, what is its average speed?"
    )
)

# Use the conversation prompt with history
history = [
    HumanMessage(content="What is the capital of France?"),
    AIMessage(content="The capital of France is Paris.")
]

conversation_result = model.invoke(
    conversation_prompt.format_messages(
        history=history,
        input="What about Germany?"
    )
)`,
          description: 'Creating and using different types of prompt templates'
        }]
      },
      {
        title: 'Memory Systems',
        content: `Memory systems allow agents to retain and utilize information across interactions. Effective memory management is crucial for maintaining context and enabling more natural, coherent conversations.

Types of memory in LangChain and LangGraph:

1. Short-Term Memory (Thread-Scoped)
   - Retains information within a single conversation thread
   - Managed as part of the agent's state
   - Persisted via thread-scoped checkpoints
   - Examples: conversation history, uploaded files, generated artifacts

2. Long-Term Memory (Cross-Thread)
   - Shared across different conversation threads
   - Organized in custom namespaces
   - Stored as JSON documents with keys
   - Examples: user preferences, learned facts, past interactions

Memory management techniques:

• Editing Message Lists
  - Removing old messages to manage context window
  - Using RemoveMessage to delete specific messages
  - Implementing custom reducers for complex operations

• Summarization
  - Condensing past conversations into summaries
  - Preserving key information while reducing token usage
  - Using LLMs to generate and update summaries

• Token-Based Truncation
  - Counting tokens in message history
  - Truncating when approaching context limits
  - Preserving critical messages (e.g., system instructions)

Learn more about memory systems in the [LangGraph documentation](https://langchain-ai.github.io/langgraph/concepts/memory/).`,
        codeExamples: [{
          typescript: `import { ChatOpenAI } from "@langchain/openai";
import { BufferMemory } from "langchain/memory";
import { ConversationChain } from "langchain/chains";
import { MessagesPlaceholder } from "@langchain/core/prompts";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StateGraph, END } from "@langchain/langgraph";
import { RunnableSequence } from "@langchain/core/runnables";
import { HumanMessage, AIMessage, RemoveMessage } from "@langchain/core/messages";
import { InMemoryStore } from "@langchain/langgraph/store";

// Simple buffer memory example
const model = new ChatOpenAI();
const memory = new BufferMemory({
  returnMessages: true,
  memoryKey: "history"
});

const prompt = ChatPromptTemplate.fromMessages([
  ["system", "You are a helpful assistant."],
  new MessagesPlaceholder("history"),
  ["human", "{input}"]
]);

const chain = RunnableSequence.from([
  {
    input: (input) => input.input,
    history: async (input) => {
      const result = await memory.loadMemoryVariables({});
      return result.history;
    }
  },
  prompt,
  model
]);

// Run the chain and save to memory
const response1 = await chain.invoke({ input: "Hello, my name is Alice." });
await memory.saveContext(
  { input: "Hello, my name is Alice." },
  { output: response1.content }
);

const response2 = await chain.invoke({ input: "What's my name?" });

// LangGraph with message management
interface GraphState {
  messages: Array<HumanMessage | AIMessage>;
}

// Function to manage message history
function manageMessageHistory(state: GraphState): GraphState {
  // Keep only the last 10 messages to manage context window
  if (state.messages.length > 10) {
    const messagesToRemove = state.messages.slice(0, state.messages.length - 10);
    return {
      messages: [
        ...messagesToRemove.map(msg => new RemoveMessage({ id: msg.id })),
        ...state.messages.slice(-10)
      ]
    };
  }
  return state;
}

// Create a graph with memory management
const workflow = new StateGraph<GraphState>({
  channels: {
    messages: {
      value: (x: Array<HumanMessage | AIMessage>) => x,
      default: () => []
    }
  }
});

// Add nodes
workflow.addNode("manage_memory", manageMessageHistory);
workflow.addNode("generate_response", async (state) => {
  const response = await model.invoke(state.messages);
  return { messages: [response] };
});

// Define edges
workflow.setEntryPoint("manage_memory");
workflow.addEdge("manage_memory", "generate_response");
workflow.addEdge("generate_response", END);

// Compile the graph
const app = workflow.compile();

// Long-term memory with InMemoryStore
const store = new InMemoryStore();
const userId = "user-123";
const namespace = [userId, "preferences"];

// Store user preferences
await store.put(
  namespace,
  "personal_info",
  {
    name: "Alice",
    location: "New York",
    interests: ["AI", "programming", "music"]
  }
);

// Retrieve user preferences
const userInfo = await store.get(namespace, "personal_info");

// Search for memories
const memories = await store.search(
  namespace,
  { query: "interests" }
);`,
          python: `from langchain_openai import ChatOpenAI
from langchain.memory import BufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableSequence
from langchain_core.messages import HumanMessage, AIMessage, RemoveMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Union, Dict, Any
from langgraph.store.memory import InMemoryStore

# Simple buffer memory example
model = ChatOpenAI()
memory = BufferMemory(
    return_messages=True,
    memory_key="history"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = RunnableSequence.from_components([
    {
        "input": lambda x: x["input"],
        "history": lambda x: memory.load_memory_variables({})["history"]
    },
    prompt,
    model
])

# Run the chain and save to memory
response1 = chain.invoke({"input": "Hello, my name is Alice."})
memory.save_context(
    {"input": "Hello, my name is Alice."},
    {"output": response1.content}
)

response2 = chain.invoke({"input": "What's my name?"})

# LangGraph with message management
class GraphState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

# Function to manage message history
def manage_message_history(state: GraphState) -> GraphState:
    # Keep only the last 10 messages to manage context window
    if len(state["messages"]) > 10:
        messages_to_remove = state["messages"][:-10]
        return {
            "messages": [
                RemoveMessage(id=msg.id) for msg in messages_to_remove
            ] + state["messages"][-10:]
        }
    return state

# Create a graph with memory management
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("manage_memory", manage_message_history)
workflow.add_node("generate_response", lambda state: {
    "messages": [model.invoke(state["messages"])]
})

# Define edges
workflow.set_entry_point("manage_memory")
workflow.add_edge("manage_memory", "generate_response")
workflow.add_edge("generate_response", END)

# Compile the graph
app = workflow.compile()

# Long-term memory with InMemoryStore
store = InMemoryStore()
user_id = "user-123"
namespace = (user_id, "preferences")

# Store user preferences
store.put(
    namespace,
    "personal_info",
    {
        "name": "Alice",
        "location": "New York",
        "interests": ["AI", "programming", "music"]
    }
)

# Retrieve user preferences
user_info = store.get(namespace, "personal_info")

# Search for memories
memories = store.search(
    namespace,
    query="interests"
)`,
          description: 'Implementing different memory systems for short-term and long-term retention'
        }]
      }
    ]
  }
};
