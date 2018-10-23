import click
import os
import tempfile

from src.config import tensorimage_path


@click.command()
@click.argument('set', required=True)
def main(set):
    with tempfile.NamedTemporaryFile(dir=tensorimage_path+'tensorimage/src/tmp/', delete=False) as temp:
        temp.write(bytes(set, 'ascii'))
        os.rename(temp.name, tensorimage_path+'tensorimage/src/tmp/opt')


main()
