# MNE-Python sandbox

This is a repository for experimental code for new techniques and ideas that may or may not make it into the official MNE-Python package. All of this is considered as work-in-progress.

## How this works
Contributions are welcome in the form of pull requests. Once the implementation of a piece of functionality is considered to be bug free and properly documented (both API docs and an example script), it can be incorporated into the master branch. Once it is in the master branch, it can be used by users of MNE-Python while the functionality awaits verification in a scientific manner (for new techniques, this means a paper). After the functionality has been verified, it can be integrated into MNE-Python.

## Code organization

If you want to add a new technique or feature, please add it as a submodule. For example, to add functionality X, arrange it in such a way that a user can do:

    import mne
    from mne_sandbox import X

to start using the functionality. The submodule may depend on the development version of MNE-Python and other submodules of MNE-sandbox. At least one example script should be placed in the `mne_sandbox/examples` folder.
