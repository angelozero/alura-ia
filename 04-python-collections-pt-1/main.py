import sys

def demonstrar_listas():
    """
    Listas são coleções MUTÁVEIS. 
    Use quando você precisa adicionar, remover ou ordenar elementos.
    """
    # 1. Criação e Adição
    frutas = ["Maçã", "Banana"]
    frutas.append("Cereja")          # Adiciona ao final
    frutas.insert(1, "Abacaxi")      # Insere em índice específico
    frutas.extend(["Manga", "Uva"])  # Merge de listas
    
    # 2. Remoção
    frutas.remove("Banana")          # Remove por valor
    item_removido = frutas.pop()     # Remove e retorna o último (ou por índice)
    del frutas[0]                    # Remove por índice via keyword
    
    # 3. Transformação e Ordenação
    numeros = [5, 2, 9, 1]
    numeros.sort()                   # In-place sort (Mais eficiente em memória)
    numeros.reverse()                # Inverte a ordem
    
    # 4. List Comprehension (Poder do Python)
    quadrados = [x**2 for x in range(5)]
    
    print(f"Lista Final: {frutas}")
    print(f"Quadrados: {quadrados}")

def demonstrar_tuplas():
    """
    Tuplas são coleções IMUTÁVEIS. 
    Use para dados que não devem mudar (ex: configurações, coordenadas, registros de BD).
    """
    # 1. Criação (Parênteses são opcionais, mas recomendados)
    ponto_3d = (10, 20, 30)
    config_db = "localhost", 5432, "admin" # Packing
    
    # 2. Unpacking (Muito usado em retornos de funções)
    host, porta, user = config_db
    
    # 3. Métodos (Apenas 2 existem, pois não alteram a estrutura)
    cores = ("vermelho", "azul", "vermelho")
    print(f"Contagem de vermelho: {cores.count('vermelho')}")
    print(f"Índice do azul: {cores.index('azul')}")
    
    # 4. Tupla de um único elemento (A vírgula é obrigatória)
    single = (1,) 

def quando_usar_cada():
    """
    Exemplos comparativos de tomada de decisão.
    """
    print("\n--- Comparação: Lista vs Tupla ---")
    
    # CASO 1: Performance e Memória
    # Tuplas são ligeiramente mais rápidas e ocupam menos memória
    lista_ex = [1, 2, 3, 4, 5]
    tupla_ex = (1, 2, 3, 4, 5)
    print(f"Tamanho Lista: {sys.getsizeof(lista_ex)} bytes")
    print(f"Tamanho Tupla: {sys.getsizeof(tupla_ex)} bytes")
    
    # CASO 2: Dicionários como Chaves
    # Apenas tuplas podem ser chaves de dicionários (por serem hashable)
    localizacao = {}
    localizacao[( -37.8136, 144.9631)] = "Melbourne CBD"
    
    # CASO 3: Semântica
    # Lista: Coleção homogênea de itens (Ex: Lista de usuários ativos)
    # Tupla: Estrutura heterogênea/Registro (Ex: (id, nome, data_nascimento))
    registro_usuario = (1, "Angelo", "2026-04-07") 
    print(f"Usuário: {registro_usuario[1]}")

def funcoes_comuns():
    """Funções built-in que funcionam em ambos (Iteráveis)"""
    data = [10, 20, 30, 40]
    
    print(f"Soma: {sum(data)}")
    print(f"Mínimo: {min(data)}")
    print(f"Máximo: {max(data)}")
    print(f"Comprimento: {len(data)}")
    
    # Enumerate: gera (índice, valor)
    for i, v in enumerate(data):
        print(f"Index {i}: {v}")

if __name__ == "__main__":
    demonstrar_listas()
    demonstrar_tuplas()
    quando_usar_cada()
    funcoes_comuns()