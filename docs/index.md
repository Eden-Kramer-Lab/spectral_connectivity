```{include} ../README.md
:relative-docs: docs/
:relative-images:
:end-before: <!-- docs/index.md skips from here: a link to the docs site itself. -->
```

```{include} ../README.md
:relative-docs: docs/
:relative-images:
:start-after: <!-- docs/index.md resumes here. -->
:end-before: <!-- docs/index.md includes this file up to this line. -->
```

```{toctree}
:caption: Guides
:hidden:
:maxdepth: 2

cookbook
choosing_a_measure
migration
llm_guide
CONNECTIVITY_METRIC_RANGES
STYLE
```

```{toctree}
:caption: Tutorials
:hidden:
:maxdepth: 2

examples/Intro_tutorial
examples/Tutorial_On_Simulated_Examples
examples/Tutorial_Using_Paper_Examples

```

```{toctree}
:caption: Reference
:hidden:
:titlesonly:
:maxdepth: 1

api
contributing
```
