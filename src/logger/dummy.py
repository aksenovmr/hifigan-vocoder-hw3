class DummyWriter:
    def __init__(
        self,
        *args,
        run_name=None,
        id_length=None,
        loss_names=None,
        log_checkpoints=False,
        mode=None,
        **kwargs,
    ):
        self.run_name = run_name
        self.step = 0

    def set_step(self, step: int):
        self.step = step

    def add_scalar(self, *args, **kwargs):
        pass

    def add_audio(self, *args, **kwargs):
        pass

    def add_histogram(self, *args, **kwargs):
        pass

    def add_images(self, *args, **kwargs):
        pass

    def close(self):
        pass