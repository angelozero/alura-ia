from factory_service import get_chat_model, get_embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def main():
    llm = get_chat_model()

    embeddings = get_embeddings()

    arquivos = [
        "27-langchain-python-criando-ferramentas-openai/data/GTB_standard_Nov23.pdf",
        "27-langchain-python-criando-ferramentas-openai/data/GTB_platinum_Nov23.pdf",
        "27-langchain-python-criando-ferramentas-openai/data/GTB_gold_Nov23.pdf",
    ]

    documentos = sum([PyPDFLoader(arquivo).load() for arquivo in arquivos], [])

    pedacos = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100
    ).split_documents(documentos)

    dados_recuperados = FAISS.from_documents(pedacos, embeddings).as_retriever(
        search_kwargs={"k": 2}
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Responda usando exclusivamente o conteudo fornecido."),
            ("human", "{input}\n\nContexto: {contexto}\n\nResposta: "),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    def responder_pergunta(pergunta):
        trechos = dados_recuperados.invoke(pergunta)
        contexto = "\n\n".join([trecho.page_content for trecho in trechos])
        resposta = chain.invoke({"input": pergunta, "contexto": contexto})
        return resposta

    print(responder_pergunta("Quais benefícios este guia traz?"))


if __name__ == "__main__":
    main()
