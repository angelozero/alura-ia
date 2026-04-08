"""
Módulo main.py: Guia de referência rápida da sintaxe Python.
Este arquivo contém exemplos práticos de tipos, coleções, controle de fluxo e POO.
"""

import math
from typing import List, Tuple, Dict, Any, Optional

# --- TIPO BÁSICOS ---
nome: str = "Angelo"
idade: int = 30
altura: float = 1.80
is_dev: bool = True

# --- STRINGS ---
"""
Operações comuns com strings para limpeza e formatação de dados.
"""
texto_bruto = "  dados_sujos_123  "
texto_limpo = texto_bruto.strip()  # Remove espaços (Trim)
minusculo = texto_limpo.lower()  # toLowerCase
substituido = texto_limpo.replace("_", " ")
partes = texto_limpo.split("_")  # Retorna lista

# --- COLEÇÕES ---

# LISTAS (Equivalente ao ArrayList)
frutas: List[str] = ["maçã", "banana"]
frutas.append("laranja")  # add()
frutas.extend(["uva", "pêra"])  # addAll()

# TUPLAS (Imutáveis)
config_fixa: Tuple[str, int] = ("localhost", 8080)

# DICIONÁRIOS (Equivalente ao HashMap)
usuario: Dict[str, Any] = {
    "id": 1,
    "login": "angelo_dev",
    "permissoes": ["admin", "user"]
}

# SETS (Valores únicos/Conjuntos)
ids_unicos = {101, 102, 101, 103}  # Resulta em {101, 102, 103}


# --- FUNÇÕES E DOCSTRINGS ---

def processar_calculo(valores: List[float], fator: float = 1.0) -> List[float]:
    """
    Realiza a multiplicação de uma lista de valores por um fator.

    Args:
        valores (list): Uma lista contendo números decimais.
        fator (float, optional): O multiplicador a ser aplicado. Default é 1.0.

    Returns:
        list: Uma nova lista com os valores processados.

    Raises:
        ValueError: Se a lista estiver vazia.
    """
    if not valores:
        raise ValueError("A lista de valores não pode estar vazia.")

    return [v * fator for v in valores]


# --- CLASSES (POO) ---

class Desenvolvedor:
    """
    Representa a entidade de um Desenvolvedor no sistema.

    Attributes:
        nome (str): Nome do profissional.
        linguagem (str): Linguagem principal de atuação.
    """

    def __init__(self, nome: str, linguagem: str):
        """Inicializa o desenvolvedor com nome e linguagem."""
        self.nome = nome
        self.linguagem = linguagem

    def saudar(self) -> str:
        """Retorna uma saudação formatada."""
        return f"Olá, eu sou {self.nome} e foco em {self.linguagem}."


# --- CONTROLE DE FLUXO E TRATAMENTO ---

def demonstracao_fluxo():
    """Exemplifica loops, condicionais e tratamento de erros."""

    # List Comprehension (Estilo Streams)
    numeros = [1, 2, 3, 4, 5]
    pares = [n for n in numeros if n % 2 == 0]

    # Try-Except-Finally
    try:
        divisao = 10 / 0
    except ZeroDivisionError as e:
        # Captura erro específico
        divisao = None
    except Exception as e:
        # Captura qualquer outro erro
        print(f"Erro inesperado: {e}")
    finally:
        # Executa independente do erro (Cleanup)
        pass

    # For-each e Enumerate
    for index, valor in enumerate(pares):
        print(f"Índice {index} tem o valor {valor}")


# --- PONTO DE ENTRADA ---

if __name__ == "__main__":
    """Execução principal para fins de teste/consulta."""
    dev = Desenvolvedor(nome, "Java/Python")
    print(dev.saudar())

    resultado = processar_calculo([10.5, 20.0], fator=2)
    print(f"Dados processados: {resultado}")
    print(f"Dicionário de usuário: {usuario}")