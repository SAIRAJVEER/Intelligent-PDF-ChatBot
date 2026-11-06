from vectordatabase import vectorDB
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# 1. Load the vector database
vectorstore = vectorDB()

# 2. Create the retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 6}
)

# 3. Define the prompt template
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful assistant who answers questions
        using the provided source documents.

        Use only the given context to answer.
        If the answer is not present in the context,
        say that you don't know.

        Keep your answer concise and use a maximum
        of three sentences.

        Context:
        {context}"""
    ),
    ("human", "{question}")
])

# 4. Initialize the Groq LLM
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

# 5. Connect the prompt to the LLM
rag_chain = prompt | llm

# 6. Ask a question
question = "What is a neural network?"

# 7. Retrieve relevant documents
docs = retriever.invoke(question)

# 8. Convert retrieved documents into context
context = "\n\n".join(
    doc.page_content for doc in docs
)

# 9. Generate the answer
response = rag_chain.invoke({
    "context": context,
    "question": question
})

print(response.content)

