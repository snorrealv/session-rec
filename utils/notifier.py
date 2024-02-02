import importlib

class Notifyer():
    def __init__(self, mode="telegram") -> None:
        self.mode = mode
        try:
            module = importlib.import_module(f"utils.notifiers.{self.mode}")
            self.notifier_instance = getattr(module, self.mode.capitalize())()
        except ModuleNotFoundError:
            raise ValueError(f"Notifier {self.mode} not found.")

    def send_message(self, message):
        self.notifier_instance.send_message(message)

    def send_exception(self, message):
        self.notifier_instance.send_exception(message)

    def send_results(self, message, results):
        self.notifier_instance.send_results(message, results)