from pydantic import BaseModel, Field


class Estudante(BaseModel):
    nome: str = Field("Nome do estudante")
    