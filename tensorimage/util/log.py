import logging


def add_coloring_to_emit_ansi(fn):
    def new(*args):
        levelno = args[1].levelno
        if levelno >= 50:
            color = '\x1b[31m'  # red
        elif levelno >= 40:
            color = '\x1b[31m'  # red
        elif levelno >= 30:
            color = '\x1b[33m'  # yellow
        elif levelno >= 20:
            color = '\x1b[32m'  # green
        elif levelno >= 10:
            color = '\x1b[35m'  # pink
        else:
            color = '\x1b[0m'  # normal
        args[1].msg = color + args[1].msg + '\x1b[0m'  # normal
        return fn(*args)

    return new


def warning(msg: str, obj, *args, **kwargs):  # Verbose level 3
    if obj.verbose >= 3:
        logging.warning(msg, *args, **kwargs)


def error(msg: str, obj, *args, **kwargs):  # Verbose level 4
    if obj.verbose >= 4:
        logging.error(msg, *args, **kwargs)


def info(msg: str, obj, *args, **kwargs):  # Verbose level 1
    if obj.verbose >= 1:
        logging.info(msg, *args, **kwargs)


def fatal(msg: str, obj, *args, **kwargs):  # Verbose level 5
    if obj.verbose >= 5:
        logging.fatal(msg, *args, **kwargs)


def debug(msg: str, obj, *args, **kwargs):  # Verbose level 2
    if obj.verbose >= 2:
        logging.debug(msg, *args, **kwargs)


def getLogger(name: str):
    return logging.getLogger(name)


def _setup():
    logging.StreamHandler.emit = add_coloring_to_emit_ansi(logging.StreamHandler.emit)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Setup logger
    formatter = logging.Formatter('%(levelname)8s %(message)s')

    hdlr = logging.StreamHandler()
    hdlr.setFormatter(formatter)
    root.addHandler(hdlr)


_setup()
