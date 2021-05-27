# 2021-msia423-Xu-Congda-assignment3

# Documenting the model pipeline with a bash script

## Build the image

```bash
 docker build -f Dockerfile_bash -t bash-example .
```

## Execute the pipeline


```bash
docker run --mount type=bind,source="$(pwd)/data",target=/app/data/ bash-example run-pipeline.sh
```

## Run tests

```bash
docker run --mount type=bind,source="$(pwd)/data",target=/app/data/ bash-example run-tests.sh
``` 