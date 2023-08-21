# RAQA (Retrieval Augmented Question Answering) + Agents
<br>


## Introduction 👋

In this project, we explore an intriguing application that interacts with the Wikipedia pages of two movies: **"Barbie"** and **"Oppenheimer"**. We also dive into their associated reviews. Our main aim is to shed light on the concept of "Agents", particularly when integrated with vast tools including large language models.

<br>

### 🌟 What are Agents? The Basics

An "Agent" is a dynamic unit, adept at accepting and processing queries. This is achieved by tapping into an expansive toolbox which may include:

- Wikipedia
- Web search functionalities
- Mathematical libraries
- Large language models like GPT-4.

Within **LangChain**, agents and chains are interlinked:

- Chains can encapsulate agents.
- Conversely, agents might harbor a chain.

What sets agents apart is their prowess in devising a sequence of real-time steps. As they sift through their toolkit to cater to a request, these steps take form. LangChain endows agents with a plethora of tools, enabling them to craft chains that resonate with the query's intricacies.

#### 🚀 Unique Strengths of Agents

Agents, embedded in the LangChain environment, boast several distinctive traits:

- **Chain-of-Thought Reasoning**: By design, agents champion chain-of-thought reasoning, ensuring context-rich responses.
- **Autonomy in Tool Selection**: They autonomously cherry-pick and sequence the most suitable tools.
- **Logic Adherence**: Agents' intrinsic logic assures lucidity in outcomes.
- **Broad Contextual Lens**: They adeptly tackle cryptic or ambiguous queries, necessitating complex reasoning or multi-hop strategies.

#### 📜 LangChain Agent's Workflow

This notebook illustrates the agent's typical workflow via a graphical representation:

1. **Entering the Chain**: Initiation of the agent's workflow.
2. **Action Input**: Based on user input, the optimal action is discerned.
3. **Observation**: Post-action outcomes are keenly observed.
4. **Thought Process**: Contemplation on the data to decide on further steps.
5. **Final Answer**: Compilation and presentation of a comprehensive answer.

<br>

### 🔍 Example with Web Search & GPT-4

```python
pip install langchain
pip install google-search-results
pip install openai

from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.llms import OpenAI
import os

os.environ['OPENAI_API_KEY'] = "YOUR_OPENAI_API_KEY"
os.environ["SERPAPI_API_KEY"] = "YOUR_SERPAPI_API_KEY"

llm = OpenAI(temperature=0, model_name='gpt-4-0314')
tools = load_tools(["serpapi", "llm-math"], llm=llm)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
response = agent.run("What year was the founder of SpaceX and Tesla born and what is the name of the first company he founded?")
print(response)
```

### 📚 Example with Wikipedia & GPT-4

```python
pip install langchain
pip install wikipedia
pip install openai

from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.llms import OpenAI
import os

os.environ['OPENAI_API_KEY'] = "YOUR_OPENAI_API_KEY"

llm = OpenAI(temperature=0, model_name='gpt-4-0314')
tools = load_tools(["wikipedia"], llm=llm)
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
response = agent.run("What year was the founder of SpaceX and Tesla born and what is the name of the first company he founded?")
print(response)
```

<br>

## Architecture 💡

### Adding reasoaning to our RAQA

<p align="center">
  <img src="/Users/salv91/Desktop/llm-engineering-ops/utils/system-designs/reasoning.png" alt="AI-Maker-Space" width="some_width">
</p>
<br>

### Ensemble Retrieval with Reciprocal Rank Fusion (RRF)

<p align="center">
  <img src="/Users/salv91/Desktop/llm-engineering-ops/utils/system-designs/ensemble retireval.png" alt="AI-Maker-Space" width="some_width">
</p>
<br>

#### **Overview:**
Ensemble retrieval uses the RRF algorithm to combine sparse and dense search results, enhancing relevant document retrieval.

#### **Key Components:**
- **Retrievers**: A set of retrievers being combined.
- **Weights**: Assigned to each retriever, defaulting to equal if unspecified.
- **Constant (c)**: Balances high and low-ranked items (default: 60).

#### **RRF Essentials:**
RRF is an unsupervised technique that merges document rankings from multiple IR systems, consistently outdoing individual systems. It calculates scores as: [RRF](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)


### Multi-query Retrieval

<p align="center">
  <img src="/Users/salv91/Desktop/llm-engineering-ops/utils/system-designs/multi query retrieval.png" alt="AI-Maker-Space" width="some_width">
</p>
<br>

<br>



### 🧠 Components Breakdown Multi-query Retrieval ML system

<br>

#### 1. 🤖 User-System Interaction:

- **Definition:** The juncture where users communicate with our RAQA (Retrieval Augmented Question Answering) system.
- **Example:** A user's query, such as "Who is King Lear?", kickstarts this interaction.

#### 2. 📊 Models:

We're harnessing the OpenAI framework for our Language Learning Model (LLM) which bifurcates into:

- **🌐 Embedding Model**:
  - **Function:** Transforms input into a compact, lower-dimensional form.
  - **Algorithm:** ADA Algorithm. 
  - **Outcome:** Embeddings (a condensed representation of input).

- **💬 Chat Model (gpt-3.5-turbo)**:
  - **Function:** Facilitates a conversational interface by discerning language intricacies.
  - **Outcome:** Replies that mimic human conversation.

#### 2. 🤖 Agents: Conversational Retrieval

LangChain provides a facile way to fashion conversational retrieval Agents.

- **How to Build**: Use the built-in function `create_retriever_tool`.
- **Tips for Efficacy**: It's imperative to give detailed natural language descriptions for the tool's purpose, ensuring optimum results.

RELEVANT DOCS:
- [create_retriever_tool](https://api.python.langchain.com/en/latest/agents/langchain.agents.agent_toolkits.conversational_retrieval.tool.create_retriever_tool.html#langchain.agents.agent_toolkits.conversational_retrieval.tool.create_retriever_tool)

#### 3. 💾 Storage: VectorStore with OpenAI Embeddings

- **Objective**: Storehouse for embeddings derived from data.
- **Harmony Note**: While there's no strict need for the embedding model to align with the LLM, it's vital to maintain congruence when embedding both the data index and the queries.

#### 4. 🔍 Embeddings Optimization: CacheBackedEmbeddings

- **Overview**: Aids in circumventing the re-embedding of similar queries, hence saving computational resources.
- **Process**: Transforms each piece of text into a unique hash. This hash then becomes the cache key, with the actual embedding as its value.
- **Distinct Traits**: 
  - Temporarily stores embeddings to prevent unnecessary recomputation.
  - Leverages the "namespace" parameter to discern between different embedding models.

#### 5. 🎬 Data Source & Processing

**IMDB Reviews of Barbie and Oppenheimer**: 
- **Origins**: Barbie movie reviews are gleaned from IMDB.
- **Action Items**: Convert this unprocessed data to be congruent with LangChain's requirements.

**Wikipedia Info on Barbie and Oppenheimer**: 
- **Objective**: Extract and utilize relevant details from Wikipedia about both Barbie and Oppenheimer for our application.


<br>

### 🛠 **Project Workflow:**
<br>

### Multi-Source Retrieval with Agent Creation in LangChain

Please refer to the [Readme](/Users/salv91/Desktop/llm-engineering-ops/projects/llm_retrieval_augmented_question_answering/README.md) for a broader overview.

This tutorial underscores the construction of an Agent and a multi-source retrieval strategy. We'll pivot from our previous endeavors and navigate the realms of multi-source retrieval systems.

#### Step 1: Multi-Source Chain Construction

🎯 **Goal**: Empower the LLM to prioritize information from most to least valuable.

🛠️ **Tool**: LangChain's "Expression Language".

🔗 **Resource**: Facing challenges? Peruse [this guide](https://python.langchain.com/docs/use_cases/question_answering/how_to/multiple_retrieval). Experimentation is the key!

```python
from langchain.prompts import ChatPromptTemplate

system_message = """Use the information from the below two sources to answer any questions.

Source 1: public user reviews about the Oppenheimer movie
<source1>
{source1}
</source1>

Source 2: the wikipedia page for the Oppenheimer movie including the plot summary, cast, and production information
<source2>
{source2}
</source2>
"""

prompt = ChatPromptTemplate.from_messages([("system", system_message), ("human", "{question}")])

oppenheimer_multisource_chain = {
    "source1": (lambda x: x["question"]) | opp_ensemble_retriever,
    "source2": (lambda x: x["question"]) | opp_ensemble_retriever,
    "question": lambda x: x["question"],
} | prompt | llm
```

#### Step 2: Setting Up the Agent Toolbelt

An Agent, akin to a superhero, needs a set of tools or abilities to operate effectively. For our Agent, these tools enable various functions.

While we're crafting rudimentary examples here, the versatility of Agents is boundless, a fact that will unveil itself as we delve deeper.

```python 
from langchain.agents import Tool

def query_oppenheimer_movie_system(question: str) -> str:
    """
    Interface with the Oppenheimer Movie Q&A system.
    
    Args:
    - question (str): A concise, well-framed question pertinent to the Oppenheimer movie.
    
    Returns:
    - str: The system's response.
    """
    return oppenheimer_multisource_chain.invoke({"question": question})

def query_barbie_information_system(question: str) -> str:
    """
    Access the Barbie Information Retrieval system.
    
    Args:
    - question (str): A comprehensive question revolving around Barbie.
    
    Returns:
    - str: The system's elucidative response.
    """
    return oppenheimer_multisource_chain.invoke({"question": question})

tools = [
    Tool(
        name="Barbie Knowledge Retriever",
        func=barbie_retriever_agent_executor.invoke,
        description="Delve into detailed insights about Barbie's history and more. Pose full-fledged questions for optimal outcomes."
    ),
    Tool(
        name="Oppenheimer Movie Query System",
        func=query_oppenheimer_movie_system,
        description="Unearth specifics about the Oppenheimer movie. Formulate comprehensive questions for precise elucidations."
    ),
]
```

#### Step 3: Power Up with LLM

After preparing our tools, we'll breathe life into our Agent with the LLM. Feel free to tinker with the prompts and find the optimal fit.

🔗 Relevant Docs: Dive into the ZeroShotAgent documentation for deeper insights.

```python 
from langchain.agents import ZeroShotAgent, AgentExecutor

prefix = """Engage in a dialogue, addressing the questions leveraging the tools at your disposal."""
suffix = """Initiate!

Question: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    prefix=prefix,
    suffix=suffix,
    tools=tools
)

from langchain import LLMChain

llm_chain = LLMChain(llm=llm, prompt=prompt)
```

#### Step 4:  Integrate the ReAct Framework

With our LLM primed, let's weave together our ZeroShotAgent and AgentExecutor, the pillars of the ReAct Agent blueprint.

🔗 Deep Dive: Familiarize yourself with the ReAct framework.

```python
barbenheimer_agent = ZeroShotAgent(
    llm_chain=llm_chain,
    tools=tools,
    verbose=True)

barbenheimer_agent_chain = AgentExecutor.from_agent_and_tools(
    agent=barbenheimer_agent,
    tools=tools,
    verbose=True)
```