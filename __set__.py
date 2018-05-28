import click
import os
import tempfile


@click.command()
@click.option('--set', required=True, help='Set operation: training, classifying, im_man')
def main(set):
    with tempfile.NamedTemporaryFile(dir='tmp/', delete=False) as temp:
        temp.write(bytes(set, 'ascii'))
        os.rename(temp.name, 'tmp/opt')


main()
