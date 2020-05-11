import os
import os.path as p

path_to_exp = p.abspath(p.join(p.dirname(p.realpath(__file__)),
                               '../../experiments/'))

def link_project(*args, dest=path_to_exp, overwrite=False):
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
    overwrite : boolean
        True/False overwrite the file (or symbolic link to file) if it exists
        in the given directory [default: False]

    Returns
    -------
    None
    """
    for arg in args:
        f = p.basename(p.normpath(arg))
        if p.isdir(dest):
            dest_i = p.join(dest, f)
        else:
            dest_i = dest
        if overwrite and p.exists(dest_i):
            os.remove(p.abspath(dest_i))
        os.symlink(p.abspath(arg), dest_i)
link_project.__doc__ = link_project.__doc__ % (path_to_exp)

def link_file(*args, file=p.join(path_to_exp,'context.py'), overwrite=False):
    """
    Create symbolic link to the `experiments/context.py` file within the given
    directory

    Parameters
    ----------
    *args : str
        directory or file path(s) where the file should be linked
    file : str
        file to be linked [default: `%s`]
    overwrite : boolean
        True/False overwrite the file (or symbolic link to file) if it exists
        in the given directory [default: False]

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
        if p.isdir(arg):
            arg = p.join(arg, f)
        if overwrite and p.exists(arg):
            os.remove(p.abspath(arg))
        os.symlink(p.abspath(file), arg)
link_file.__doc__ = link_file.__doc__ % (p.join(path_to_exp,'context.py'))
