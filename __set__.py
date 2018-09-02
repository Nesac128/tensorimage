import click
import os
import tempfile


@click.command()
@click.argument('set', required=True)
def main(set):
    with tempfile.NamedTemporaryFile(dir='tmp/', delete=False) as temp:
        temp.write(bytes(set, 'ascii'))
        os.rename(temp.name, 'tmp/opt')


main()
