import { Module } from '../../../types';

export const mentalModelsModule: Module = {
  id: 'mental-models',
  title: 'Mental Models',
  description: 'Essential patterns for thinking about agent systems',
  duration: '1.5 hours',
  level: 'intermediate',
  content: {
    sections: [
      {
        title: 'ReAct Pattern',
        content: `The ReAct (Reasoning + Acting) pattern is a powerful framework for agent decision-making that combines reasoning and acting in an iterative process. Learn more about ReAct in the [documentation](https://python.langchain.com/docs/modules/agents/agent_types/react).

Key Components:

1. Thought
   - Internal reasoning process
   - Situation analysis
   - Strategy formation

2. Action
   - Tool selection
   - Parameter preparation
   - Execution

3. Observation
   - Result analysis
   - Error handling
   - Context update

4. Iteration
   - Continuous improvement
   - Goal tracking
   - Success evaluation

Benefits:
• Clear reasoning chains
• Traceable decisions
• Better error handling
• Improved accuracy`,
        codeExamples: [{
          typescript: `import { ChatOpenAI } from "@langchain/openai";
import { createReactAgent, AgentExecutor } from "langchain/agents";
import { Calculator, WebBrowser } from "@langchain/community/tools";
import { pull } from "langchain/hub";

// Initialize the model
const model = new ChatOpenAI({
  modelName: "gpt-4",
  temperature: 0
});

// Define tools
const tools = [
  new Calculator(),
  new WebBrowser()
];

// Create a ReAct agent with custom prompt
const agent = await createReactAgent({
  llm: model,
  tools,
  prompt: await pull("hwchase17/react-complex")
});

const executor = new AgentExecutor({
  agent,
  tools,
  verbose: true,
  maxIterations: 5,
  returnIntermediateSteps: true
});

// Execute with detailed output
const result = await executor.invoke({
  input: "Research the latest fusion energy breakthrough and calculate its efficiency"
});

// Access intermediate steps
result.intermediateSteps.forEach(step => {
  console.log("Thought:", step.thought);
  console.log("Action:", step.action);
  console.log("Observation:", step.observation);
});`,
          python: `from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.tools import Calculator, WebBrowser
from langchain.hub import pull

# Initialize the model
model = ChatOpenAI(
    model_name="gpt-4",
    temperature=0
)

# Define tools
tools = [
    Calculator(),
    WebBrowser()
]

# Create a ReAct agent with custom prompt
agent = create_react_agent(
    llm=model,
    tools=tools,
    prompt=pull("hwchase17/react-complex")
)

executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5,
    return_intermediate_steps=True
)

# Execute with detailed output
result = executor.invoke({
    "input": "Research the latest fusion energy breakthrough and calculate its efficiency"
})

# Access intermediate steps
for step in result["intermediate_steps"]:
    print("Thought:", step.thought)
    print("Action:", step.action)
    print("Observation:", step.observation)`,
          description: 'Implementing a ReAct agent with detailed step tracking'
        }]
      }
    ]
  }
};