import { Module } from '../../../types';

export const buildingBlocksModule: Module = {
  id: 'building-blocks',
  title: 'Building Blocks for Agents',
  description: 'Understanding the essential components for creating AI agents',
  duration: '2 hours',
  level: 'intermediate',
  content: {
    sections: [
      {
        title: 'Tools and Tool Usage',
        content: `Tools are functions that agents can use to interact with external systems and perform specific tasks. They are a fundamental building block that extends an agent's capabilities beyond just generating text.

Key concepts about tools:

• Definition: A tool associates a Python/JavaScript function with a schema that defines the function's name, description, and expected arguments
• Purpose: Tools allow agents to take actions in the world, access external data, and perform computations
• Integration: Tools can be passed to chat models that support tool calling, enabling models to request specific function executions

Types of tools:

1. Built-in Tools
   - Web browsers for searching and retrieving information
   - Calculators for mathematical operations
   - Shell commands for system interactions
   - File operations for reading/writing data

2. Custom Tools
   - Domain-specific functions
   - API integrations
   - Database queries
   - Specialized utilities

Creating tools:

• Using the @tool decorator (Python) or tool function (JavaScript)
• Defining clear names, descriptions, and argument schemas
• Implementing proper error handling
• Supporting both synchronous and asynchronous execution

Learn more about tools in the [LangChain documentation](https://python.langchain.com/docs/concepts/tools/).`,
        codeExamples: [{
          typescript: `import { tool } from "@langchain/core/tools";
import { z } from "zod";

// Create a weather tool with Zod schema validation
const getWeather = tool(
  async ({ location }) => {
    // In a real implementation, this would call a weather API
    if (location.toLowerCase().includes("london")) {
      return "Rainy, 15°C";
    } else if (location.toLowerCase().includes("sahara")) {
      return "Sunny, 45°C";
    }
    return "Partly cloudy, 22°C";
  },
  {
    name: "get_weather",
    description: "Get the current weather for a specific location",
    schema: z.object({
      location: z.string().describe("The city and country, e.g., London, UK"),
    }),
  }
);

// Create a calculator tool
const calculator = tool(
  async ({ expression }) => {
    try {
      // Using Function constructor for safe evaluation
      // Note: In production, use a more secure method
      return new Function("return " + expression)().toString();
    } catch (error) {
      return "Error calculating result: " + error.message;
    }
  },
  {
    name: "calculator",
    description: "Evaluate a mathematical expression",
    schema: z.object({
      expression: z.string().describe("The mathematical expression to evaluate"),
    }),
  }
);

// Create a tool that returns both content and artifacts
const generateChart = tool(
  async ({ data, chartType }) => {
    // In a real implementation, this would generate a chart
    const chartUrl = "https://example.com/chart.png";

    // Return both a message for the model and the chart URL as an artifact
    return {
      content: "I've generated a " + chartType + " chart based on your data.",
      artifacts: [{ type: "image", url: chartUrl }]
    };
      },
      {
        name: "generate_chart",
        description: "Generate a chart from data",
        schema: z.object({
          data: z.string().describe("The data for the chart in CSV format"),
          chartType: z.enum(["bar", "line", "pie"]).describe("The type of chart to generate"),
        }),
      }
);

// Using the tools directly
const weatherResult = await getWeather.invoke({ location: "London, UK" });
console.log(weatherResult); // "Rainy, 15°C"

const calculationResult = await calculator.invoke({ expression: "23 * 7" });
console.log(calculationResult); // "161"

// Inspecting tool schemas
console.log(getWeather.name); // "get_weather"
console.log(getWeather.description); // "Get the current weather for a specific location"
console.log(getWeather.schema); // The Zod schema object`,
          python: `from langchain_core.tools import tool
from typing import Dict, Any, Tuple, List, Optional
from typing_extensions import Annotated

# Create a weather tool with type annotations
@tool
def get_weather(location: Annotated[str, "The city and country, e.g., London, UK"]) -> str:
    """Get the current weather for a specific location."""
    # In a real implementation, this would call a weather API
    if "london" in location.lower():
        return "Rainy, 15°C"
    elif "sahara" in location.lower():
        return "Sunny, 45°C"
    return "Partly cloudy, 22°C"

# Create a calculator tool
@tool
def calculator(expression: Annotated[str, "The mathematical expression to evaluate"]) -> str:
    """Evaluate a mathematical expression."""
    try:
        # Using eval for demonstration - in production use a safer method
        return str(eval(expression))
    except Exception as e:
        return f"Error calculating result: {str(e)}"

# Create a tool that returns both content and artifacts
@tool(return_type="content_and_artifact")
def generate_chart(
    data: Annotated[str, "The data for the chart in CSV format"],
    chart_type: Annotated[str, "The type of chart to generate (bar, line, pie)"]
) -> Tuple[str, Dict[str, Any]]:
    """Generate a chart from data."""
    # In a real implementation, this would generate a chart
    chart_url = "https://example.com/chart.png"

    # Return both a message for the model and the chart URL as an artifact
    return (
        f"I've generated a {chart_type} chart based on your data.",
        {"type": "image", "url": chart_url}
    )

# Using the tools directly
weather_result = get_weather.invoke({"location": "London, UK"})
print(weather_result)  # "Rainy, 15°C"

calculation_result = calculator.invoke({"expression": "23 * 7"})
print(calculation_result)  # "161"

# Inspecting tool schemas
print(get_weather.name)  # "get_weather"
print(get_weather.description)  # "Get the current weather for a specific location"
print(get_weather.args)  # The JSON schema for the tool's arguments`,
          description: 'Creating and using tools with schemas and validation'
        }]
      },
      {
        title: 'Retrieval Systems',
        content: `Retrieval systems allow agents to access and utilize external knowledge, making them more capable and accurate. These systems efficiently identify relevant information from large datasets in response to queries.

Key concepts about retrieval:

• Purpose: Retrieval systems ground agent responses in factual information, reducing hallucinations
• Process: Query analysis → Information retrieval → Integration with agent reasoning
• Types: Vector stores, lexical search, SQL databases, graph databases, and hybrid approaches

Retrieval components:

1. Document Loaders
   - Load data from various sources (files, websites, databases)
   - Support multiple formats (text, PDFs, HTML, images)
   - Handle different storage systems (local, cloud, APIs)

2. Text Splitters
   - Break long documents into manageable chunks
   - Balance chunk size for context preservation and retrieval granularity
   - Support different splitting strategies (by character, token, semantic units)

3. Embedding Models
   - Convert text into vector representations
   - Capture semantic meaning in high-dimensional space
   - Enable similarity-based retrieval

4. Vector Stores
   - Store and index vector embeddings
   - Support efficient similarity search
   - Offer metadata filtering and hybrid search capabilities

5. Retrievers
   - Provide a unified interface for different retrieval systems
   - Accept a query string and return relevant documents
   - Implement various retrieval strategies (similarity, MMR, hybrid)

Advanced retrieval techniques:

• Query transformation: Rewriting queries to improve retrieval performance
• Self-query: Using LLMs to generate metadata filters and search terms
• Multi-query retrieval: Generating multiple search queries for better coverage
• Hypothetical Document Embeddings (HyDE): Using synthetic documents to improve search

Learn more about retrieval in the [LangChain documentation](https://python.langchain.com/docs/concepts/retrieval/).`,
        codeExamples: [{
          typescript: `import { ChatOpenAI } from "@langchain/openai";
import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";

// 1. Load documents
const loader = new CheerioWebBaseLoader(
  "https://docs.smith.langchain.com/overview"
);
const docs = await loader.load();

// 2. Split text into chunks
const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 500,
  chunkOverlap: 50
});
const splitDocs = await textSplitter.splitDocuments(docs);

// 3. Create embeddings and store in vector database
const embeddings = new OpenAIEmbeddings();
const vectorStore = await MemoryVectorStore.fromDocuments(
  splitDocs,
  embeddings
);

// 4. Create a retriever
const retriever = vectorStore.asRetriever({
  k: 4, // Number of documents to retrieve
  searchType: "similarity" // Can be "similarity", "mmr", etc.
});

// 5. Create a model and prompt
const model = new ChatOpenAI({
  modelName: "gpt-3.5-turbo",
  temperature: 0
});

const prompt = ChatPromptTemplate.fromTemplate(\`
Answer the question based only on the following context:
{context}

Question: {question}
\`);

// 6. Create a chain that combines documents
const chain = await createStuffDocumentsChain({
  llm: model,
  prompt,
  outputParser: new StringOutputParser()
});

// 7. Create a RAG chain
const ragChain = retriever.pipe(chain);

// 8. Run the chain
const response = await ragChain.invoke({
  question: "What is LangSmith used for?"
});

console.log(response);

// Advanced retrieval techniques
// Example of query transformation
const queryTransformer = ChatPromptTemplate.fromTemplate(\`
Given the user question below, reformulate it to make it a standalone question
that will help retrieve relevant documents from a vector database.

User question: {question}

Reformulated question:
\`).pipe(model).pipe(new StringOutputParser());

const betterQuery = await queryTransformer.invoke({
  question: "What can it do?"
});

// Use the transformed query for retrieval
const docsFromBetterQuery = await retriever.invoke(betterQuery);`,
          python: `from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import CheerioWebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain

# 1. Load documents
loader = CheerioWebBaseLoader(
    "https://docs.smith.langchain.com/overview"
)
docs = loader.load()

# 2. Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
split_docs = text_splitter.split_documents(docs)

# 3. Create embeddings and store in vector database
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings
)

# 4. Create a retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4},  # Number of documents to retrieve
    search_type="similarity"  # Can be "similarity", "mmr", etc.
)

# 5. Create a model and prompt
model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0
)

prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:
{context}

Question: {question}
""")

# 6. Create a chain that combines documents
chain = create_stuff_documents_chain(
    llm=model,
    prompt=prompt,
    document_variable_name="context",
    output_parser=StrOutputParser()
)

# 7. Create a RAG chain
rag_chain = create_retrieval_chain(
    retriever,
    chain
)

# 8. Run the chain
response = rag_chain.invoke({
    "question": "What is LangSmith used for?"
})

print(response["answer"])

# Advanced retrieval techniques
# Example of query transformation
query_transformer = (
    ChatPromptTemplate.from_template("""
Given the user question below, reformulate it to make it a standalone question
that will help retrieve relevant documents from a vector database.

User question: {question}

Reformulated question:
""")
    | model
    | StrOutputParser()
)

better_query = query_transformer.invoke({
    "question": "What can it do?"
})

# Use the transformed query for retrieval
docs_from_better_query = retriever.invoke(better_query)`,
          description: "Building a Retrieval-Augmented Generation (RAG) system with document loading, chunking, embedding, and retrieval"
        }]
      },
      {
        title: 'Structured Output',
        content: `Structured output is a technique that allows models to return data in specific formats, making it easier to process and use in downstream applications. This is essential for agents that need to interact with databases, APIs, or other structured systems.

Key concepts about structured output:

• Purpose: Ensure model outputs conform to expected schemas and formats
• Benefits: Improved reliability, easier validation, and seamless integration with other systems
• Methods: Tool calling, JSON mode, and output parsing

Structured output approaches:

1. Schema Definition
   - Using JSON Schema to define output structure
   - Leveraging Pydantic (Python) or Zod (TypeScript) for type validation
   - Defining clear field descriptions and constraints

2. Tool Calling
   - Binding schemas as tools to models
   - Models decide when to use tools based on the task
   - Extracting structured data from tool calls

3. JSON Mode
   - Enforcing JSON output format directly
   - Supported by specific model providers
   - Useful for simple structured outputs

4. Output Parsing
   - Converting raw model outputs to structured formats
   - Handling parsing errors gracefully
   - Implementing retry strategies for failed parses

Best practices:

• Use descriptive field names and clear descriptions
• Keep schemas as simple as possible
• Implement proper error handling for parsing failures
• Validate outputs against schemas

Learn more about structured output in the [LangChain documentation](https://python.langchain.com/docs/concepts/structured_outputs/).`,
        codeExamples: [{
          typescript: `import { ChatOpenAI } from "@langchain/openai";
import { z } from "zod";
import { StructuredOutputParser } from "langchain/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";

// 1. Define a schema using Zod
const movieSchema = z.object({
  title: z.string().describe("The title of the movie"),
  director: z.string().describe("The director of the movie"),
  year: z.number().describe("The year the movie was released"),
  genres: z.array(z.string()).describe("List of genres for the movie"),
  rating: z.number().min(0).max(10).describe("Rating from 0-10"),
  summary: z.string().describe("Brief summary of the movie plot")
});

// 2. Create a model
const model = new ChatOpenAI({
  modelName: "gpt-4o",
  temperature: 0
});

// 3. Method 1: Using with_structured_output
const structuredModel = model.withStructuredOutput(movieSchema);

// Create a prompt
const prompt = ChatPromptTemplate.fromMessages([
  ["system", "Extract movie information from the user's text."],
  ["human", "{input}"]
]);

// Create a chain
const chain = prompt.pipe(structuredModel);

// Run the chain
const result = await chain.invoke({
  input: "I watched The Matrix last night. It's a sci-fi movie from 1999 directed by the Wachowskis. It's about a computer programmer who discovers the world is a simulation. I'd rate it 9/10."
});

console.log(result);
// {
//   title: "The Matrix",
//   director: "The Wachowskis",
//   year: 1999,
//   genres: ["Sci-Fi", "Action"],
//   rating: 9,
//   summary: "A computer programmer discovers the world is a simulation."
// }

// 4. Method 2: Using JSON mode
const jsonModel = model.withStructuredOutput({
  method: "jsonMode"
});

const jsonResult = await jsonModel.invoke(
  "Return a JSON object with the names and ages of the three main characters in Harry Potter."
);

console.log(jsonResult);
// {
//   "characters": [
//     {"name": "Harry Potter", "age": 11},
//     {"name": "Hermione Granger", "age": 11},
//     {"name": "Ron Weasley", "age": 11}
//   ]
// }

// 5. Method 3: Using output parsers
const parser = StructuredOutputParser.fromZodSchema(movieSchema);

const parserPrompt = ChatPromptTemplate.fromMessages([
  ["system", \`Extract movie information from the user's text.
  \${parser.getFormatInstructions()}\`],
  ["human", "{input}"]
]);

const parserChain = parserPrompt.pipe(model).pipe(parser);

const parserResult = await parserChain.invoke({
  input: "Inception (2010) is a mind-bending thriller directed by Christopher Nolan. It features Leonardo DiCaprio as a thief who steals information by infiltrating dreams. I'd give it 8.5/10."
});

console.log(parserResult);`,
          python: `from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

# 1. Define a schema using Pydantic
class Movie(BaseModel):
    title: str = Field(description="The title of the movie")
    director: str = Field(description="The director of the movie")
    year: int = Field(description="The year the movie was released")
    genres: List[str] = Field(description="List of genres for the movie")
    rating: float = Field(description="Rating from 0-10", ge=0, le=10)
    summary: str = Field(description="Brief summary of the movie plot")

# 2. Create a model
model = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)

# 3. Method 1: Using with_structured_output
structured_model = model.with_structured_output(Movie)

# Create a prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract movie information from the user's text."),
    ("human", "{input}")
])

# Create a chain
chain = prompt | structured_model

# Run the chain
result = chain.invoke({
    "input": "I watched The Matrix last night. It's a sci-fi movie from 1999 directed by the Wachowskis. It's about a computer programmer who discovers the world is a simulation. I'd rate it 9/10."
})

print(result)
# Movie(
#   title="The Matrix",
#   director="The Wachowskis",
#   year=1999,
#   genres=["Sci-Fi", "Action"],
#   rating=9.0,
#   summary="A computer programmer discovers the world is a simulation."
# )

# 4. Method 2: Using JSON mode
json_model = model.with_structured_output(method="json_mode")

json_result = json_model.invoke(
    "Return a JSON object with the names and ages of the three main characters in Harry Potter."
)

print(json_result)
# {
#   "characters": [
#     {"name": "Harry Potter", "age": 11},
#     {"name": "Hermione Granger", "age": 11},
#     {"name": "Ron Weasley", "age": 11}
#   ]
# }

# 5. Method 3: Using output parsers
parser = PydanticOutputParser(pydantic_object=Movie)

parser_prompt = ChatPromptTemplate.from_messages([
    ("system", f"Extract movie information from the user's text.\n{parser.get_format_instructions()}"),
    ("human", "{input}")
])

parser_chain = parser_prompt | model | parser

parser_result = parser_chain.invoke({
    "input": "Inception (2010) is a mind-bending thriller directed by Christopher Nolan. It features Leonardo DiCaprio as a thief who steals information by infiltrating dreams. I'd give it 8.5/10."
})

print(parser_result)`,
          description: "Implementing structured output using different methods: with_structured_output, JSON mode, and output parsers"
        }]
      },
      {
        title: 'Multimodality',
        content: `Multimodality refers to the ability to work with different types of data beyond just text, such as images, audio, and video. This capability is becoming increasingly important for building more versatile and capable AI agents.

Key concepts about multimodality:

• Definition: The ability to process and generate multiple types of data
• Applications: Image analysis, audio processing, document understanding, and more
• Integration: Combining different modalities for richer interactions

Multimodal capabilities:

1. Input Modalities
   - Text: Natural language input
   - Images: Photos, diagrams, charts, screenshots
   - Audio: Speech, music, sounds
   - Video: Motion, temporal information
   - Documents: PDFs, presentations, spreadsheets

2. Processing Approaches
   - Single-model multimodality: Using models that natively support multiple modalities
   - Multi-model pipelines: Combining specialized models for different modalities
   - Cross-modal translation: Converting between different modalities

3. Output Modalities
   - Text generation: Descriptions, analyses, summaries
   - Image generation: Creating or editing visual content
   - Audio generation: Speech synthesis, sound effects

Implementation considerations:

• Model selection: Choose models that support your required modalities
• Data representation: Format inputs appropriately for each modality
• Performance: Consider latency and resource requirements
• Integration: Ensure smooth transitions between modalities

Learn more about multimodality in the [LangChain documentation](https://python.langchain.com/docs/concepts/multimodality/).`,
        codeExamples: [{
          typescript: `import { ChatOpenAI } from "@langchain/openai";
import { ChatAnthropic } from "@langchain/anthropic";
import { HumanMessage } from "@langchain/core/messages";

// Initialize multimodal models
const openAIVisionModel = new ChatOpenAI({
  modelName: "gpt-4o",
  temperature: 0
});

const anthropicVisionModel = new ChatAnthropic({
  modelName: "claude-3-opus-20240229",
  temperature: 0
});

// Example 1: Analyzing an image with OpenAI
async function analyzeImageOpenAI(imageUrl: string, prompt: string) {
  const message = new HumanMessage({
    content: [
      {
        type: "text",
        text: prompt
      },
      {
        type: "image_url",
        image_url: {
          url: imageUrl
        }
      }
    ]
  });

  const response = await openAIVisionModel.invoke([message]);
  return response.content;
}

// Example 2: Analyzing an image with Anthropic
async function analyzeImageAnthropic(imageUrl: string, prompt: string) {
  const message = new HumanMessage({
    content: [
      {
        type: "text",
        text: prompt
      },
      {
        type: "image",
        source: {
          type: "url",
          url: imageUrl
        }
      }
    ]
  });

  const response = await anthropicVisionModel.invoke([message]);
  return response.content;
}

// Example 3: Analyzing multiple images
async function analyzeMultipleImages(imageUrls: string[], prompt: string) {
  const content = [
    {
      type: "text",
      text: prompt
    }
  ];

  // Add all images to the content array
  for (const url of imageUrls) {
    content.push({
      type: "image_url",
      image_url: {
        url: url
      }
    });
  }

  const message = new HumanMessage({ content });
  const response = await openAIVisionModel.invoke([message]);
  return response.content;
}

// Example 4: Document analysis (PDF)
async function analyzePDF(pdfUrl: string, prompt: string) {
  const message = new HumanMessage({
    content: [
      {
        type: "text",
        text: prompt
      },
      {
        type: "image_url",
        image_url: {
          url: pdfUrl
        }
      }
    ]
  });

  const response = await openAIVisionModel.invoke([message]);
  return response.content;
}

// Usage examples
const imageUrl = "https://example.com/image.jpg";
const chartUrl = "https://example.com/chart.png";
const pdfUrl = "https://example.com/document.pdf";

// Analyze a single image
const imageAnalysis = await analyzeImageOpenAI(
  imageUrl,
  "Describe what you see in this image in detail."
);

// Compare two images
const imageComparison = await analyzeMultipleImages(
  [imageUrl, chartUrl],
  "Compare these two images and tell me the key differences."
);

// Extract information from a PDF
const pdfAnalysis = await analyzePDF(
  pdfUrl,
  "Summarize the key points from this document."
);`,
          python: `from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

# Initialize multimodal models
openai_vision_model = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)

anthropic_vision_model = ChatAnthropic(
    model="claude-3-opus-20240229",
    temperature=0
)

# Example 1: Analyzing an image with OpenAI
def analyze_image_openai(image_url: str, prompt: str):
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": prompt
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url
                }
            }
        ]
    )

    response = openai_vision_model.invoke([message])
    return response.content

# Example 2: Analyzing an image with Anthropic
def analyze_image_anthropic(image_url: str, prompt: str):
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": prompt
            },
            {
                "type": "image",
                "source": {
                    "type": "url",
                    "url": image_url
                }
            }
        ]
    )

    response = anthropic_vision_model.invoke([message])
    return response.content

# Example 3: Analyzing multiple images
def analyze_multiple_images(image_urls: list, prompt: str):
    content = [
        {
            "type": "text",
            "text": prompt
        }
    ]

    # Add all images to the content list
    for url in image_urls:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": url
            }
        })

    message = HumanMessage(content=content)
    response = openai_vision_model.invoke([message])
    return response.content

# Example 4: Document analysis (PDF)
def analyze_pdf(pdf_url: str, prompt: str):
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": prompt
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": pdf_url
                }
            }
        ]
    )

    response = openai_vision_model.invoke([message])
    return response.content

# Usage examples
image_url = "https://example.com/image.jpg"
chart_url = "https://example.com/chart.png"
pdf_url = "https://example.com/document.pdf"

# Analyze a single image
image_analysis = analyze_image_openai(
    image_url,
    "Describe what you see in this image in detail."
)

# Compare two images
image_comparison = analyze_multiple_images(
    [image_url, chart_url],
    "Compare these two images and tell me the key differences."
)

# Extract information from a PDF
pdf_analysis = analyze_pdf(
    pdf_url,
    "Summarize the key points from this document."
)`,
          description: "Working with multimodal data including images and documents using different model providers"
        }]
      }
    ]
  }
};
