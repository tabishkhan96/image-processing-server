FROM python:3.9.6-slim AS python-base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    PYSETUP_PATH="/opt/pysetup" \
    VENV_PATH="/opt/pysetup/.venv"

ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

FROM python-base AS poetry-build

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        curl \
        build-essential

ENV POETRY_VERSION=1.1.7
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -

WORKDIR $PYSETUP_PATH
COPY ./poetry.lock ./pyproject.toml ./
RUN poetry install --no-dev && find $VENV_PATH \
    \( -type d -a -name test -o -name tests \) \
    -o \( -type f -a -name '*.pyc' -o -name '*.pyc' \) \
    -exec rm -rf '{}' \+


FROM python-base AS server
ENV FASTAPI_ENV=production

# EXIFTOOL
RUN apt-get update && apt-get install --no-install-recommends -y libimage-exiftool-perl jpegoptim

COPY --from=poetry-build $VENV_PATH $VENV_PATH
COPY ./docker/entrypoint.sh /entrypoint.sh
COPY ./api /app/api
COPY ./docker/server_start.sh /app
COPY ./files /app/files

RUN chmod +x /entrypoint.sh && chmod +x /app/server_start.sh

WORKDIR /app

ENTRYPOINT ["/entrypoint.sh"]
CMD [ "./server_start.sh" ]