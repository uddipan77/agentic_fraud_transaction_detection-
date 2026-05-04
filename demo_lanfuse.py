import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import ulid
from langfuse import Langfuse, observe
from langfuse.langchain import CallbackHandler #connects LangChain model calls to Langfuse

load_dotenv()

#This creates the LangChain model wrapper.
#Langfuse will observe calls made through this client using the callback handler.
model = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=50,
)

#This connects your program to the Langfuse backend.
langfuse_client = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://challenges.reply.com/langfuse")
)


def generate_session_id():
    team = os.getenv("TEAM_NAME", "tutorial").replace(" ", "-")
    return f"{team}-{ulid.new().str}"

"""
"callbacks": [langfuse_handler]

This is very important.

This tells LangChain:

whenever this model call happens,
notify Langfuse through the callback handler

That is how the actual model generation gets captured in Langfuse.
"""

def invoke_langchain(model, prompt, langfuse_handler, session_id):
    messages = [HumanMessage(content=prompt)]
    response = model.invoke(messages, config={
        "callbacks": [langfuse_handler],
        "metadata": {"langfuse_session_id": session_id},
    })
    return response.content


@observe()
def run_llm_call(session_id, model, prompt):
    langfuse_handler = CallbackHandler()
    return invoke_langchain(model, prompt, langfuse_handler, session_id)


def main():
    questions = [
        "What is machine learning?",
        "Explain neural networks briefly.",
        "What is the difference between AI and ML?"
    ]

    session_id = generate_session_id()

    for i, question in enumerate(questions, 1):
        response = run_llm_call(session_id, model, question)
        print(f"[{i}/{len(questions)}] {question} -> {response[:60]}...")

    langfuse_client.flush()

    print(f"\n{len(questions)} traces sent | session: {session_id}")
    print("Check the Langfuse dashboard to verify (may take a few minutes to update).")


if __name__ == "__main__":
    main()
