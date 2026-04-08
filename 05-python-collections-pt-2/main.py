import sys

def demonstrar_dicionarios():
    """
    Dicionários (dict) são coleções de pares Chave:Valor.
    Ideal para mapeamentos, caches e representação de objetos (JSON-like).
    """
    # 1. Criação e Acesso
    dev = {
        "nome": "Angelo",
        "linguagem": "Python",
        "experiencia": 10
    }
    
    # Acesso seguro (evita KeyError)
    stack = dev.get("stack", "Não definida") 
    
    # 2. Adição e Atualização
    dev["cidade"] = "Melbourne"
    dev.update({"status": "Ativo", "experiencia": 11})
    
    # 3. Remoção
    removido = dev.pop("status") # Remove e retorna o valor
    del dev["experiencia"]      # Remove via keyword
    
    # 4. Iteração Eficiente
    print("--- Chaves e Valores ---")
    for chave, valor in dev.items():
        print(f"{chave.upper()}: {valor}")

    # 5. Dictionary Comprehension
    quadrados_dict = {x: x**2 for x in range(5)}
    print(f"Dicionário de Quadrados: {quadrados_dict}")

def demonstrar_conjuntos():
    """
    Conjuntos (set) são coleções NÃO ORDENADAS de elementos ÚNICOS.
    Ideal para remover duplicatas e operações matemáticas de conjuntos.
    """
    # 1. Criação
    numeros = {1, 2, 2, 3, 4, 4} # {1, 2, 3, 4} - Duplicatas ignoradas
    linguagens = set(["Python", "Java", "Go", "Python"])
    
    # 2. Operações de Conjunto (Onde o Set brilha)
    set_a = {1, 2, 3, 4}
    set_b = {3, 4, 5, 6}
    
    uniao = set_a | set_b           # Ou set_a.union(set_b) -> {1, 2, 3, 4, 5, 6}
    interseccao = set_a & set_b     # Ou set_a.intersection(set_b) -> {3, 4}
    diferenca = set_a - set_b       # Elementos em A que não estão em B -> {1, 2}
    diff_simetrica = set_a ^ set_b  # Elementos em apenas um dos conjuntos -> {1, 2, 5, 6}
    
    # 3. Adição e Remoção
    set_a.add(10)
    set_a.discard(99) # discard não gera erro se o item não existir (ao contrário de .remove())

def quando_usar_cada():
    """Comparação técnica de casos de uso."""
    
    # CASO 1: Remoção de Duplicatas (O clássico)
    lista_suja = ["melbourne", "sydney", "melbourne", "brisbane"]
    lista_limpa = list(set(lista_suja))
    
    # CASO 2: Teste de Pertencimento (Membership Test)
    # Em listas, isso é O(n). Em sets/dicts, é O(1).
    emails_bloqueados = {"spam@test.com", "bot@web.com"}
    email_atual = "user@gmail.com"
    
    if email_atual in emails_bloqueados: # Extremamente rápido mesmo com milhões de itens
        print("Acesso negado.")

    # CASO 3: Dicionário como 'Switch Case' (Padrão comum em Python)
    def acao_save(): return "Salvando..."
    def acao_delete(): return "Excluindo..."
    
    switch = {
        "SAVE": acao_save,
        "DELETE": acao_delete
    }
    
    resultado = switch.get("SAVE", lambda: "Ação Inválida")()
    print(f"Resultado Switch: {resultado}")

if __name__ == "__main__":
    print("=== EXECUTANDO DICIONÁRIOS ===")
    demonstrar_dicionarios()
    
    print("\n=== EXECUTANDO CONJUNTOS ===")
    demonstrar_conjuntos()
    
    quando_usar_cada()