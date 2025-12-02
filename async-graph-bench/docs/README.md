# ReadMe

## Building the Docs locally

```bash
sphinx-build -b html docs/source docs/build
```

## Generating API Docs

Modify `generate_api.py` to contain all nodes that are to be included in the api section, then run the script. This will replace already existing documentation markdown files under `api/`