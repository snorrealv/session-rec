
import importlib

class Exporter():
    def __init__(self, mode="soak") -> None:
        self.mode = mode
        try:
            module = importlib.import_module(f"utils.notifiers.{self.mode}")
            self.exporter_instanse = getattr(module, self.mode.capitalize())()
        except ModuleNotFoundError:
            raise ValueError(f"Exporter {self.mode} not found.")

    def export(self, file_path):
        self.exporter_instanse.export(file_path)