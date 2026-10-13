import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI


# Carga las variables del archivo .env al entorno interno de python
load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite",temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un asistente útil."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | llm

history = []

print("Chat en terminal (escribe 'salir' para terminar)\n")

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

    respuesta = chain.invoke({"history": history, "input": user_input})
    print(f"Asistente: {respuesta.content}")

    # Actualizamos el historial
    history.extend([
        HumanMessage(content=user_input),
        AIMessage(content=respuesta.content)
    ])
