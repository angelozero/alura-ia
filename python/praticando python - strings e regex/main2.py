import re


def validar_email(email):
    # Expressão regular para um e-mail padrão
    padrao = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

    if re.match(padrao, email):
        return True
    return False


# Teste
email_teste = "contato@exemplo.com.br"
print(f"O e-mail é válido? {validar_email(email_teste)}")
email_teste = "contato-exemplo.com.br"
print(f"O e-mail é válido? {validar_email(email_teste)}")