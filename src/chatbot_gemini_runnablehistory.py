import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

# Carga las variables del archivo .env al entorno interno de python
load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite",temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un asistente útil."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | llm

store = {}

def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Cadena con memoria automatica por sesion
chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

print("Chat en terminal (escribe 'salir' para terminar)\n")

session_id = "sesion_terminal"

while True:
    try:
        user_input = input("Tú: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nHasta Luego!")
        break

    if not user_input:
        continue
    if user_input.lower() in {"salir","exit","quit"}:
        print("Hasta Luego!")
        break

    respuesta = chain_with_memory.invoke(
        {"input": user_input},
        config={"configurable": {"session_id":session_id}}
        )
    print(f"Asistente: {respuesta.content}")
    
