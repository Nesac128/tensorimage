import click
import os
import tempfile

from config import nnir_path


@click.command()
@click.argument('set', required=True)
def main(option):
    with tempfile.NamedTemporaryFile(dir=nnir_path+'nnir/src/tmp/', delete=False) as temp:
        temp.write(bytes(option, 'ascii'))
        os.rename(temp.name, nnir_path+'nnir/src/tmp/opt')


main()
