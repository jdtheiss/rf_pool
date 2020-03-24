import os
import os.path as p

path_to_exp = p.abspath(p.join(p.dirname(p.realpath(__file__)),
                               '../../experiments/'))

def link_project(*args, dest=path_to_exp):
    """
    Create symbolic link to a project folder or jupyter notebook within the
    `experiments` directory

    Parameters
    ----------
    *args : str
        file/directory path(s) to project(s) that should be linked
    dest : str
        destination path where a symbolic link is made
        [default: `%s`]

    Returns
    -------
    None
    """
    for arg in args:
        f = p.basename(p.normpath(arg))
        os.symlink(p.abspath(arg), p.join(path_to_exp, f))
link_project.__doc__ = link_project.__doc__ % (path_to_exp)

def link_file(*args, file=p.join(path_to_exp,'context.py'), overwrite=False):
    """
    Create symbolic link to the `experiments/context.py` file within the given
    directory

    Parameters
    ----------
    *args : str
        directory path(s) where the file should be linked
    file : str
        file to be linked [default: `%s`]
    overwrite : boolean
        True/False overwrite the file (or symbolic link to file) if it exists
        in the given directory

    Returns
    -------
    None

    Notes
    -----
    Linking the `context.py` file is useful for importing a specific version of
    rf_pool outside its relative path. Once linked, the rf_pool version relative
    to the `context.py` file can be loaded via `from context import rf_pool`.
    """
    f = p.basename(p.normpath(file))
    for arg in args:
        if overwrite and p.exists(p.join(arg, f)):
            os.remove(p.join(arg, f))
        os.symlink(p.abspath(file), p.join(arg, f))
link_file.__doc__ = link_file.__doc__ % (p.join(path_to_exp,'context.py'))
