def main(dict_data):
    print(dict_data)
    print(f"KEY: {dict_data.keys()}")
    print(f"VALUE: {dict_data.values()}")
    print(f"ITEMS: {dict_data.items()}")
    print(f"LENGTH: {len(dict_data)}")
    print("")
    print(f"O nome é: {dict_data.get("nome")}")
    print("")

if __name__ == "__main__":
    dict0 = dict(nome="Angelo", idade=20)
    dict1 = {
        "nome": "Jake",
        "idade": 20,
        "endereco": ["SP", "RJ", "MG"]
    }

    main(dict0)
    main(dict1)
