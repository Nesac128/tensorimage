import click
import os
import tempfile

from src.config import nnir_path


@click.command()
@click.argument('set', required=True)
def main(set):
    with tempfile.NamedTemporaryFile(dir=nnir_path+'nnir/src/tmp/', delete=False) as temp:
        temp.write(bytes(set, 'ascii'))
        os.rename(temp.name, nnir_path+'nnir/src/tmp/opt')


main()
