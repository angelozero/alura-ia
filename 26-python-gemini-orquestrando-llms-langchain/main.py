from agent_orchestrator import AgentOrchestrator

SEPARATOR = "=" * 60

def main():
    print(f"\n{SEPARATOR}")
    print("🚀 [MAIN] Iniciando aplicação de análise de imagens com LangChain")
    print(SEPARATOR)

    print("\n📦 [MAIN] Criando AgentOrchestrator...")
    orchestrator = AgentOrchestrator()
    print("✅ [MAIN] AgentOrchestrator criado com sucesso")

    query = "Faça uma análise da imagem macaco.png"
    print(f"\n💬 [MAIN] Enviando query ao agente: '{query}'")
    print(SEPARATOR)

    response = orchestrator.agent.invoke({"messages": [{"role": "user", "content": query}]})

    print(f"\n{SEPARATOR}")
    print("🏁 [MAIN] Execução concluída")
    messages = response.get("messages", [])
    final_message = messages[-1] if messages else None
    if final_message:
        print(f"📝 [MAIN] Resposta final do agente:\n{final_message.content}")
    print(SEPARATOR)

if __name__ == "__main__":
    main()