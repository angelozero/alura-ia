from abc import ABC, abstractmethod

# 1. Single Responsibility Principle (SRP) - Princípio da Responsabilidade Única
# Uma classe deve ter apenas um motivo para mudar.
class Order:
    def __init__(self, items):
        self.items = items

    def calculate_total(self):
        return sum(item['price'] for item in self.items)

class OrderRepository:
    def save(self, order):
        print(f"Salvando pedido no banco de dados...")


# 2. Open/Closed Principle (OCP) - Princípio Aberto/Fechado
# Entidades de software devem estar abertas para extensão, mas fechadas para modificação.
class DiscountStrategy(ABC):
    @abstractmethod
    def apply(self, total):
        pass

class TenPercentDiscount(DiscountStrategy):
    def apply(self, total):
        return total * 0.9

class NoDiscount(DiscountStrategy):
    def apply(self, total):
        return total


# 3. Liskov Substitution Principle (LSP) - Princípio da Substituição de Liskov
# Objetos de uma superclasse devem ser substituíveis por objetos de suas subclasses 
# sem quebrar a aplicação.
class PaymentProcessor(ABC):
    @abstractmethod
    def process(self, amount):
        pass

class CreditCardProcessor(PaymentProcessor):
    def process(self, amount):
        print(f"Processando R${amount} via Cartão de Crédito.")

class PixProcessor(PaymentProcessor):
    def process(self, amount):
        print(f"Processando R${amount} via PIX.")


# 4. Interface Segregation Principle (ISP) - Princípio da Segregação de Interface
# Uma classe não deve ser forçada a implementar interfaces que não utiliza.
# Em Python, usamos classes abstratas menores em vez de uma única "gorda".
class Refundable(ABC):
    @abstractmethod
    def refund(self, amount):
        pass

class StandardPayment(PaymentProcessor):
    # Processador comum que não aceita reembolso
    def process(self, amount):
        print(f"Pagamento de R${amount} realizado.")

class PremiumPayment(PaymentProcessor, Refundable):
    # Processador que implementa pagamento E reembolso
    def process(self, amount):
        print(f"Pagamento Premium de R${amount} realizado.")

    def refund(self, amount):
        print(f"Reembolsando R${amount}...")


# 5. Dependency Inversion Principle (DIP) - Princípio da Inversão de Dependência
# Dependa de abstrações, não de implementações concretas.
class OrderService:
    def __init__(self, repository: OrderRepository, payment_processor: PaymentProcessor):
        # O serviço não sabe se o processador é Pix ou Crédito, ele depende da interface.
        self.repository = repository
        self.payment_processor = payment_processor

    def checkout(self, order, discount: DiscountStrategy):
        total = order.calculate_total()
        total_with_discount = discount.apply(total)
        
        self.payment_processor.process(total_with_discount)
        self.repository.save(order)


# --- EXECUÇÃO (MAIN) ---
if __name__ == "__main__":
    # Setup
    meu_pedido = Order([{'name': 'Laptop', 'price': 5000}, {'name': 'Mouse', 'price': 200}])
    repo = OrderRepository()
    
    # Podemos trocar o processador facilmente (DIP e LSP)
    processador_vendas = PixProcessor()
    estrategia_desconto = TenPercentDiscount()

    # Injeção de Dependência
    servico = OrderService(repo, processador_vendas)
    
    print("Iniciando Finalização de Compra:")
    servico.checkout(meu_pedido, estrategia_desconto)