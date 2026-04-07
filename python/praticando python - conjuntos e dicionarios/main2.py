def main(dict_data):
    dict_data["endereco"].remove("MG")
    dict_data["endereco"].append("BH")

    print(dict_data)


if __name__ == "__main__":
    dict1 = {
        "nome": "Jake",
        "idade": 20,
        "endereco": ["SP", "RJ", "MG"]
    }

    main(dict1)
