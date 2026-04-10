from abc import ABC, abstractmethod

# --- 1. INTERFACES (ISP & LSP) ---

class NotificationStrategy(ABC):
    """Interface para o Strategy Pattern (OCP/LSP)"""
    @abstractmethod
    def send(self, message: str, receiver: str):
        pass

class Subscriber(ABC):
    """Interface para o Observer Pattern"""
    @abstractmethod
    def update(self, message: str):
        pass


# --- 2. IMPLEMENTAÇÕES CONCRETAS (SRP & Strategy Pattern) ---

class EmailNotification(NotificationStrategy):
    def send(self, message: str, receiver: str):
        print(f"Enviando E-mail para {receiver}: {message}")

class SMSNotification(NotificationStrategy):
    def send(self, message: str, receiver: str):
        print(f"Enviando SMS para {receiver}: {message}")


# --- 3. FACTORY METHOD (OCP & Encapsulamento) ---

class NotificationFactory:
    """Encapsula a criação de objetos, facilitando a extensão (OCP)"""
    @staticmethod
    def create_notification(type: str) -> NotificationStrategy:
        if type == "email":
            return EmailNotification()
        elif type == "sms":
            return SMSNotification()
        raise ValueError("Tipo de notificação desconhecido")


# --- 4. OBSERVERS (ISP) ---

class Logger(Subscriber):
    def update(self, message: str):
        print(f"[LOG]: Registro de atividade: {message}")

class AdminAlert(Subscriber):
    def update(self, message: str):
        print(f"[ALERTA]: Notificando administrador sobre: {message}")


# --- 5. CLASSE PRINCIPAL (DIP & Observer Pattern) ---

class NotificationService:
    """
    DIP: Depende de abstrações (NotificationStrategy e Subscriber).
    SRP: Gerencia apenas o fluxo de notificações e inscritos.
    """
    def __init__(self):
        self._subscribers = []

    def attach(self, subscriber: Subscriber):
        self._subscribers.append(subscriber)

    def _notify_subscribers(self, message: str):
        for subscriber in self._subscribers:
            subscriber.update(message)

    def send_notification(self, strategy: NotificationStrategy, message: str, receiver: str):
        # Executa o envio
        strategy.send(message, receiver)
        # Notifica os interessados (Observer)
        self._notify_subscribers(f"Notificação enviada para {receiver}")


# --- 6. EXECUÇÃO (MAIN) ---

if __name__ == "__main__":
    # Inicializa o serviço e os observadores
    service = NotificationService()
    service.attach(Logger())
    service.attach(AdminAlert())

    # Usa a Factory para criar as estratégias (OCP)
    try:
        email_sender = NotificationFactory.create_notification("email")
        sms_sender = NotificationFactory.create_notification("sms")

        print("--- Rodada 1: E-mail ---")
        service.send_notification(email_sender, "Bem-vindo ao SOLID!", "usuario@dev.com")

        print("\n--- Rodada 2: SMS ---")
        service.send_notification(sms_sender, "Seu código é 1234", "1199999999")

    except Exception as e:
        print(f"Erro: {e}")