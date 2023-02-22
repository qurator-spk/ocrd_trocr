import click

from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor
from ocrd_trocr.recognize import TrOcrRecognize


@click.command()
@ocrd_cli_options
def ocrd_trocr_recognize(*args, **kwargs):
    """
    Run TrOCR recognition.
    """
    return ocrd_cli_wrap_processor(TrOcrRecognize, *args, **kwargs)
