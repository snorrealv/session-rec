class Soak():
    def __init__(self) -> None:
        self.Running = True

    def __getattr__(self, name):
        def soak(*args, **kwargs):
            print("soaked")
        return soak
