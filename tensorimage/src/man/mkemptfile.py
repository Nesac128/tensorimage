def mkemptfile(path):
    with open(path, 'a') as f:
        if 'json' in path:
            f.write('{}')
        f.close()
