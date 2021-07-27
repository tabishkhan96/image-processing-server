#!/bin/sh
uvicorn --host=0.0.0.0 --port=8080 --factory api.main:create_app --workers $(nproc --all)