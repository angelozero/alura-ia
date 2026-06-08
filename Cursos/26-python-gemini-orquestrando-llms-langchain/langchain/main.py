from factory import get_chat_model
from langchain_core.messages import HumanMessage
from my_helper import encode_image


def run_hello_world():
    print("--- Iniciando comunicação com a IA ---")

    try:

        # 1. Instancia o modelo usando a configuração do seu arquivo original
        llm = get_chat_model()

        # 2. Define a mensagem de teste
        imagem = encode_image(
            "26-python-gemini-orquestrando-llms-langchain/data/macaco.png"
        )
        pergunta = "Descreva a imagem sendo o mais simples possível."

        messages = [
            HumanMessage(
                content=[
                    {"type": "text", "text": pergunta},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{imagem}"},
                    },
                ]
            )
        ]

        # 3. Chama a IA
        print("Enviando mensagem para a IA... ")
        response = llm.invoke(messages)

        # 4. Exibe o resultado
        print("--- Resposta da IA ---")
        print(response.content)
        print("-----------------------")

    except Exception as e:
        print(f"Erro ao conectar com o LiteLLM: {e}")


if __name__ == "__main__":
    run_hello_world()
