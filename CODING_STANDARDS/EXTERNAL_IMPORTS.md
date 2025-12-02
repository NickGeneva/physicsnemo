<!-- markdownlint-disable-->


# EXTERNAL_IMPORTS - Coding Standards

## Overview

This document outlines coding standards for managing external imports in
physicsnemo.  The motivation of these rules is to ensure our code base remains
usable and changes in one area do not break another, unrelated area by
introducing external dependencies that are not supported.

**Important:** These rules are enforced as strictly as possible. Deviations
from these standards should only be made when absolutely necessary and must be
documented with clear justification in code comments and approved during code
review.

PhysicsNeMo dependencies are contained in pyproject.toml.  For all code in
physicsnemo's python package, or tests, this is the **sole source of truth**
for external dependencies.  Examples may introduce additional dependencies
for a specific example via requirements.txt.

Physicsnemo is organized hierarchally, meaning submodules import acyclicly
and therefore dependencies grow in higher levels of physicsnemo.  For example,
the dependency list for `physicsnemo.core` is a subset of the dependency list
of `physicsnemo.utils`, which is a subset of `physicsnemo.nn`, etc.  To
enforce this hierarchy, the dependencies are organized by "dependency groups".
See PEP 735 for more information.  In `pyproject.toml`, we organize
by dependency group and include lower level groups in higher level groups.

Not every component of physicsnemo that requires specific external dependencies
rises to the level of introducing that as a hard dependency.  For some examples,
we use `cuml` in kNN calculations, and `torch_geometric` and `torch_scatter` in
some GNN models.  These packages can be more complicated to install, and
therefore aren't part of the default installation list.  They instead appear either
in an `-extras` group or as an optional dependency list.  

Every external import in physics nemo must fall into one of two categories:

1) It is an official, hard dependency of the submodule or a lower-level submodule
   where it is imported.  Examples are `torch`, `warp`, etc.
2) It is not an official hard dependency.  It is listed in the extras or in
   an optional dependency group.  It's import is protected (see below) to
   prevent installation and runtime errors unless that code is actively used.


## Protecting imports

There are two patterns in use to protect imports, depending on the specific
use case.

### Locally Necessary Imports

For some components, the external package is so critical that functionality
can not be delivered without it.  For example, PyG in graph models.

In this case, the user should follow a delayed-error pattern:

1. Developers should check availability of a package with `physicsnemo.utils.version_check`
   as a soft-check (no exceptions raised).
2. Inside an `if:` scope, if the dependency is available, import it via 
   `importlib.import_module` and implement the feature needed.
3. Inside an `else:` scope, when the dependency is not available, implement
   the same names as exposed in the `if` path with no functionality.  Instead,
   for functions immediately raise an exception on call.  For classes, raise
   an exception on instantiation.  For classes with static methods, treat them
   as functions.

**Raised exceptions about missing imports should be informative**.  Do not just
state "package X is missing".  Instead provide information about the raiser, the
package, and the installation steps.  For example:


```

import importlib

import torch

from physicsnemo.core.version_check import check_version_spec

CUML_AVAILABLE = check_version_spec("cuml", "24.0.0", hard_fail=False)
CUPY_AVAILABLE = check_version_spec("cupy", "13.0.0", hard_fail=False)

if CUML_AVAILABLE and CUPY_AVAILABLE:
    cuml = importlib.import_module("cuml")
    cp = importlib.import_module("cupy")
    
    def knn_impl(points, queries, k) -> torch.Tensor:
       ....

else:

    def knn_impl(* args, **kwargs) -> None:
        """
        Dummy implementation for when cuml is not available.
        """

        raise ImportError(
            "physics nemo kNN: cuml or cupy is not installed, can not be used as a backend for a knn search"
            "Please install cuml and cupy, for installation instructions see: https://docs.rapids.ai/install"
        )

```

Though this introduces 


PACKAGE_AVAILALBE = 


### Locally Optional Imports

Some packages offer an accelerated/improved execution path at the cost of an additional
dependency.  For these packages, the implementation should include a reference
implementation and an additional execution path.  To avoid full reuse of code,
several patterns are acceptable, depending on the circumstance.

1. Runtime dispatch at module-level.  In this path, the dependency is a core
   component of the implementation, and an entry-point function will allow users
   to select a "backend" for execution of their code.  There should be an "auto"
   path that selects (or attempts to select) the best path.  The default path
   must use standard dependencies.  The additional paths must still protect imports
   and not raise an error unless specifically selected an the import fails.  In
   this pattern, the different backend implementations should live in separate
   files.  Example: `physicsnemo.nn.neighbors`
2. Runtime dispatch at file-level.  In this path, the dependency is a small
   component of the implementation and it is more reasonable to select a dispatch
   path at the python execution level automatically.  Implementations should
   live all in one file, and developers can perform a soft check (no `raise`)
   with `physicsnemo.core.version_check`.  The boolean result of this check
   can be used to automatically select execution paths in the code, or it may
   be used with user-settings to raise an error at runtime if unavailable.