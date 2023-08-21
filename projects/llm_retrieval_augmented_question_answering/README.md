<h1 align="center">Retrieval Augmented Question Answering</h1>


## Introduction 👋

This project focuses on creating a question-answering system tailored for the 2023 Barbie movie reviews. By utilizing the power of Retrieval Augmented Question Answering (RAQA) techniques, our model will not only understand the nuances in the questions posed but also efficiently retrieve the most relevant information from a sea of reviews.
<br>
### What is Retrieval Augmented Question Answering (RAQA)?
<br>
Retrieval Augmented Question Answering is an advanced approach in the realm of Natural Language Processing (NLP). Traditional Question Answering (QA) models predict answers based solely on the provided context (e.g., a paragraph or an article). In contrast, RAQA extends this by adding a retrieval step, where the model searches for and fetches relevant external information to better answer the question.

The primary advantage of RAQA is that it doesn't need all potential information to be loaded into its memory. Instead, it efficiently retrieves the necessary pieces of information from a vast database or collection of documents as and when needed, leading to potentially more accurate and informed answers.

<br>

## Architecture 💡

<p align="center">
  <img src="/Users/salv91/Desktop/llm-engineering-ops/utils/system-designs/retrieval_augmented_questions_answering.png" alt="AI-Maker-Space" width="some_width">
</p>
<br>

### 🧠 Components Breakdown:

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


#### 3. 💾 Storage: VectorStore with OpenAI Embeddings

- **Purpose:** A vault for the embeddings.
- **Consistency Note:** While the embedding model doesn't need to mirror the LLM, consistency is pivotal when embedding both our index and queries.

#### 4. 🔍 Embeddings Optimization: CacheBackedEmbeddings

- **Description:** An optimizer to sidestep redundant embedding of akin queries, thereby conserving computational grunt.
- **Mechanism:** Each text piece gets morphed into a hash, which then acts as the cache key with the embedding as the associated value.
- **Unique Features:** 
  - Can transiently save embeddings to eschew re-computation.
  - Utilizes the "namespace" parameter to differentiate between various embedding models.

#### 5. 🎬 Data: IMDB Reviews of Barbie

- **Source:** Reviews of the Barbie movie are scooped from IMDB.
- **Immediate To-Dos:** Morph this raw data into a LangChain-compatible format.



<br>

### 🛠 **Project Workflow:**
<br>

#### **1. 📦 Data Collection**
- **Description:** After gathering our review data into a loader, the next step is to segment these reviews for ease of processing.
- **Tools Used:** We're using the `selenium` Python package for this purpose.

#### **2. 🔄 Data Preparation** 
- **Objective:** Segmenting text into smaller chunks.
- **Tool:** The `RecursiveCharacterTextSplitter`.
- **Importance:** Proper text segmentation can greatly impact an application's performance.
- **Documentation:** [RecursiveCharacterTextSplitter](https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/recursive_text_splitter)

#### **3. 📂 Creating Index**
- **Understanding 'Index':** It signifies structured documents, made ready for queries, retrieval, and integration in the LLM application stack.

#### **4. 🔍 Selecting Vector Store** 
- **Objective:** Integrate our VectorStore with the OpenAI embeddings model.
- **Consistency Note:** The choice of embedding model should align during both the index and queries embedding phase.
- **Efficiency Tip:** The `CacheBackedEmbeddings` helps avoid redundant embedding tasks. [Details](https://python.langchain.com/docs/modules/data_connection/caching_embeddings)

#### **5. 📖 Ensuring Source Integrity**
- **Method:** We use `return_source_documents=True` to maintain the original source of each review.
- **Challenge with LLM:** LLMs can often hallucinate. Supplying response sources ensures end-user transparency.
- **Research References on Hallucinations:** 
  - [Link1](https://arxiv.org/abs/2202.03629)
  - [Link2](https://arxiv.org/abs/2305.15852)
  - [Link3](https://arxiv.org/abs/2303.16104)
  - [Link4](https://arxiv.org/abs/2305.18248)
  - [Study on Reducing Hallucination](https://arxiv.org/pdf/2104.07567.pdf)

#### **6. 🤖 Selecting LLM** 
- **Model Choice:** Our LLM pick is the `gpt-3.5-turbo` model for the RetrievalQAWithSources chain.
- **Documentation:** Here's the [`OpenAIChat()` documentation](https://python.langchain.com/docs/modules/model_io/models/chat/) for guidance.

#### **7. ⛓ Building a Retrieval Chain**
- **Objective:** Design a Retrieval Chain for semantic question posing over our dataset.
- **Simplicity with LangChain:** Though LangChain offers a user-friendly interface, delving deeper into its documentation and resources is always beneficial.

#### **8. 🚀 Deployment**
- **Description:** Once the application is refined, it's time to deploy it. We'll containerize the application using Docker and deploy it to Hugging Face Spaces.
- **Tool Used:** Chainlin
- **Steps:** 
  1. **Containerization:** Use Docker to containerize the application.
  2. **Pushing the Docker Image:** Push the Docker image to the Hugging Face Container Registry.
  3. **Deployment:** Finally, deploy the application on Hugging Face Spaces.
- **Details (TODO):** 
   - Build, test, and push the Docker image to Hugging Face's container registry.
   - Deploy the image to Hugging Face Spaces through the platform's UI.

<br>