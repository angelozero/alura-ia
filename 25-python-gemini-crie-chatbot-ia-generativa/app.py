import os
import uuid

from flask import Flask, render_template, request, Response
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from helper import carrega, salva
from selecionar_persona import personas, selecionar_persona
from gerenciar_historico import remover_mensagens_mais_antigas
from gerenciar_imagem import gerar_imagem_base64, get_image_media_type
from factory import get_chat_model

load_dotenv()

app = Flask(__name__)
app.secret_key = 'alura'

contexto = carrega("dados/musimart.txt")

caminho_imagem_enviada = None
UPLOAD_FOLDER = "imagens_temporarias"

# Initialise the LLM once from factory (reads MODEL_NAME / BASE_URL / API_KEY from .env)
llm = get_chat_model()

# Conversation history stored as a list of LangChain message objects
historico: list = []

def _build_system_prompt(personalidade: str) -> str:
    return f"""
    # PERSONA

    Você é um chatbot de atendimento a clientes de um e-commerce. 
    Você não deve responder perguntas que não sejam dados do ecommerce informado!

    Você deve utilizar apenas dados que estejam dentro do 'contexto'

    # CONTEXTO
    {contexto}

    # PERSONALIDADE
    {personalidade}

    # Histórico
    Acesse sempre o históricio de mensagens, e recupere informações ditas anteriormente.
    """


def bot(prompt: str) -> str:
    global caminho_imagem_enviada, historico

    maximo_tentativas = 1
    repeticao = 0

    while True:
        try:
            personalidade = personas[selecionar_persona(prompt)]
            system_prompt = _build_system_prompt(personalidade)

            # Build the user message — with or without an image
            if caminho_imagem_enviada:
                image_data = gerar_imagem_base64(caminho_imagem_enviada)
                media_type = get_image_media_type(caminho_imagem_enviada)

                user_message = HumanMessage(
                    content=[
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_data}"
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                f"{prompt}\n"
                                "Utilize as caracteristicas da imagem em sua resposta."
                            ),
                        },
                    ]
                )

                os.remove(caminho_imagem_enviada)
                caminho_imagem_enviada = None
            else:
                user_message = HumanMessage(content=prompt)

            historico.append(user_message)

            # Trim history when it grows too large (keep last 10 messages)
            if len(historico) > 10:
                historico = remover_mensagens_mais_antigas(historico)

            messages = [SystemMessage(content=system_prompt)] + historico

            resposta = llm.invoke(messages)

            historico.append(AIMessage(content=resposta.content))

            return resposta.content

        except Exception as erro:
            repeticao += 1
            if repeticao >= maximo_tentativas:
                return "Erro no LLM: %s" % erro

            if caminho_imagem_enviada:
                os.remove(caminho_imagem_enviada)
                caminho_imagem_enviada = None


@app.route("/upload_imagem", methods=["POST"])
def upload_imagem():
    global caminho_imagem_enviada

    if "imagem" in request.files:
        imagem_enviada = request.files["imagem"]
        nome_arquivo = str(uuid.uuid4()) + os.path.splitext(imagem_enviada.filename)[1]

        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        caminho_arquivo = os.path.join(UPLOAD_FOLDER, nome_arquivo)
        imagem_enviada.save(caminho_arquivo)
        caminho_imagem_enviada = caminho_arquivo
        return "Imagem enviada com sucesso", 200

    return "Nenhum arquivo enviado", 400


@app.route("/chat", methods=["POST"])
def chat():
    prompt = request.json["msg"]
    resposta = bot(prompt)
    return resposta


@app.route("/")
def home():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
