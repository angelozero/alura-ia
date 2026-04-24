from factory import get_chat_model
from my_helper import encode_image
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser


def run_hello_world():
    print("--- Iniciando comunicação com a IA ---")

    try:

        # 1. Instancia o modelo usando a configuração do seu arquivo original
        llm = get_chat_model()

        # 2. Define a mensagem de teste
        imagem = encode_image(
            "/Users/angelo/Projects/AI/alura-ia/26-python-gemini-orquestrando-llms-langchain/data/macaco.png"
        )

        # 3. Cria o prompt usando o ChatPromptTemplate
        template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content="""
                            Você é um assistente de descrição de imagens. Responda de forma simples e direta.

                            # Formato de Saída Esperado:
                            Descrição: 'Coloque aqui a descrição da imagem'
                            Rótulo: 'Coloque aqui uma lista com três termos chaves separados por virgula'
                        """
                ),
                HumanMessagePromptTemplate.from_template(
                    [
                        {
                            "type": "text",
                            "text": "Descreva a imagem sendo o mais simples possível.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,{imagem_informada}"
                            },
                        },
                    ]
                ),
            ]
        )

        chain = template | llm | StrOutputParser()

        # 4. Chama a IA
        print("Enviando mensagem para a IA... ")
        response = chain.invoke({"imagem_informada": imagem})

        # 5. Exibe o resultado
        print("--- Resposta da IA ---")
        print(response)
        print("-----------------------")

    except Exception as e:
        print(f"Erro ao conectar com o LiteLLM: {e}")


if __name__ == "__main__":
    run_hello_world()
