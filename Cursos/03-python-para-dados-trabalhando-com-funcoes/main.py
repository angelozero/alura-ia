def main():
    # Sua função lambda original
    metodo_lambda = lambda x, y, z: (x * 2) + (y * 3) + (z * 4)

    # --- LIST COMPREHENSION ---
    numeros = [1, 2, 3, 4, 5]
    quadrados = [n**2 for n in numeros]
    
    # --- DICT COMPREHENSION ---
    
    # 1. Mapeando números aos seus quadrados {chave: valor}
    mapa_quadrados = {n: n**2 for n in numeros}
    print(f"Dicionário de quadrados: {mapa_quadrados}")

    # 2. Criando um dicionário a partir de uma lista de nomes (Nome: Tamanho)
    nomes = ["Alice", "Bruno", "Caio"]
    tamanho_nomes = {nome: len(nome) for nome in nomes}
    print(f"Tamanho dos nomes: {tamanho_nomes}")

    # 3. Dict Comprehension com sua Lambda
    # Usando o índice como chave e o resultado da lambda como valor
    trios = [(1, 1, 1), (2, 2, 2), (3, 3, 3)]
    resultados_dict = {f"Trio_{i}": metodo_lambda(x, y, z) for i, (x, y, z) in enumerate(trios)}
    print(f"Dicionário via lambda: {resultados_dict}")

if __name__ == "__main__":
    main()