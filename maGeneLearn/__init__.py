# maGeneLearn/__init__.py

__version__ = "0.1.0"

# If you want to allow `import maGeneLearn; maGeneLearn.cli`:
from .cli import cli  

__all__ = ["cli", "__version__"]
