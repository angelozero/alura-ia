"""
Módulo de exemplo de boas práticas PEP 8 e Tipagem Estática.
Este arquivo demonstra a implementação de um sistema de processamento de dados.
"""

import logging
from typing import List, Dict, Optional, Union, Final
from abc import ABC, abstractmethod

# Constantes em UPPER_CASE (PEP 8)
TIMEOUT_LIMIT: Final = 30


class BaseProcessor(ABC):
    """Classe Abstrata seguindo o princípio da responsabilidade única."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def process(self, data: Union[str, Dict]) -> bool:
        """Assinatura de método obrigatória para subclasses."""
        pass


class DataValidator:
    """Helper class para validação de esquemas."""

    @staticmethod
    def is_valid(payload: Dict) -> bool:
        # Métodos estáticos não precisam de 'self'
        return "id" in payload


class DocumentProcessor(BaseProcessor):
    """
    Implementação concreta de um processador de documentos.
    Demonstra o uso de Type Hinting e tratamento de exceções.
    """

    def __init__(self, name: str, version: float = 1.0):
        super().__init__(name)
        self.version = version
        self._internal_cache: List[Dict] = []  # Atributo "protegido"

    def process(self, data: Union[str, Dict]) -> bool:
        """
        Processa os dados recebidos. 
        Note o uso de Union para múltiplos tipos de entrada.
        """
        if isinstance(data, dict):
            if DataValidator.is_valid(data):
                self._internal_cache.append(data)
                return True
        return False

    def get_first_entry(self) -> Optional[Dict]:
        """Retorna o primeiro item ou None se a lista estiver vazia."""
        try:
            return self._internal_cache[0]
        except IndexError:
            logging.warning(f"Processador {self.name} está vazio.")
            return None


def run_pipeline(processors: List[BaseProcessor], raw_data: List[Dict]) -> int:
    """
    Função principal que demonstra iteração e tipagem de listas.
    Sempre use 4 espaços para indentação (PEP 8).
    """
    success_count = 0
    
    for processor in processors:
        for item in raw_data:
            if processor.process(item):
                success_count += 1
                
    return success_count


if __name__ == "__main__":
    # Configuração básica de logging
    logging.basicConfig(level=logging.INFO)

    # Dados de exemplo
    data_stream = [
        {"id": 1, "content": "RAG pipeline setup"},
        {"user": "unknown"},  # Inválido (sem ID)
        {"id": 2, "content": "LLM orchestration"}
    ]

    # Instanciando objetos
    doc_proc = DocumentProcessor(name="MainProcessor")
    
    # Execução
    total = run_pipeline(processors=[doc_proc], raw_data=data_stream)
    
    print(f"Processamento concluído. Itens válidos: {total}")
    print(f"Primeiro registro: {doc_proc.get_first_entry()}")