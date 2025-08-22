# contimod_graphene: Helper package for contimod

> **Note:** Our package does not currently include a documentation. *


## Installation

### Option 1a (for developers)

You can `git clone` the repository, activate your python environment of choice and install the package as editable using
```bash
pip install -e .
```
This will allow you to use `import contimod_graphene as cm_graphene` in your python code. 
You can uninstall the package any with `pip uninstall contimod_graphene`.

If you plan to contribute to the package, you must learn how to use `git` and how to create pull requests.

### Option 1b (for developers)

We now use [`hatch`](https://github.com/pypa/hatch) as dev tool, which you need to install seperately. It automates the entire development process. A suitable environment can for example be created using
```bash
hatch env create
hatch shell
```

### Option 2 (preferred method for users)

If you are sure that you will not need to modify the package, then open the terminal and run
```bash
pip install git+https://github.com/skilledwolf/contimod_graphene.git
```
This will allow you to do `import contimod as cm` in your python code. You can uninstall the package with `pip uninstall contimod_graphene`.

### Option 3 (cross-platform)

*Note: This method is not tested on all platforms. It may or may not fail on arm-based systems (such as Apple Silicon).*

Third-party requirements:
 - Docker
 - repo2docker

To spin up a containerized jupyter environment, run:
```bash
$ jupyter-repo2docker https://github.com/skilledwolf/contimod_graphene.git
```

While this is the fastest way to spin up a container, you can also containerize this package yourself.

## Credit 
This package is developed and maintained by Dr. Tobias Wolf. Feel free to contact me, and please give me credit if you use this work. 
