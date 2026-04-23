from langchain_core.prompts import ChatPromptTemplate
from factory import get_chat_model

PROMPT_TEMPLATE = """
Responda a pergunta baseada apenas no seguinte contexto:
{context}

---

Responda a pergunta baseada apenas no seguinte contexto:
{question}
"""

def main():

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context="", question="Ola, qual é a capital da França?")

    
    llm = get_chat_model()
    response = llm.invoke(prompt)

    
    print(f"\nResposta:\n{response.content}\n\n")


if __name__ == "__main__":
    main()
