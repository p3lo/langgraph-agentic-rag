from dotenv import load_dotenv

load_dotenv()

from graph.graph import app

if __name__ == "__main__":
    print("Hello from agentic-rag!")
    result = app.invoke(input={"question": "explain agent memory"})
    print(result["generation"])
