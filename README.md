## Image Processing Server

HTTP microservice for working with images. It includes all the basic image processing functions: crop/resize/filters/rotate/blur/compress/noise and etc. To upload image to the server, you need to use multipart/form-data (no base64). The project is still in development and I will also add new features in the future.

The project was made using FastAPI and Pillow.

## Installation

```shell
git clone https://github.com/aquarokk/image-processing-server
cd image-processing-server
# install deps
poetry install
# or wihout dev deps
poetry install --no-dev
# run tests
pytest
# run server
uvicorn --host localhost --port 8080 --factory api.main:create_app --workers 4
```

To check the application status, navigate to http://localhost:8080/about

## Run project using Docker

```shell
# Clone repo
git clone https://github.com/aquarokk/image-processing-server
cd image-processing-server
# Build image
docker build -t my-image-server:latest --file docker/Dockerfile .
# Create and start a container
docker container create --name <container_name> -p 8080:8080 my-image-server:latest && docker container start <container_name>
```

To check the application status, navigate to http://localhost:8080/about

## Here is an example of a client request to the server:

```python
import asyncio
import os
import time

import aiofiles
import aiohttp

# Images for test
files = [
    os.path.join("../tests/images", path)
    for path in os.listdir("../tests/images")
    if path.endswith((".png", ".jpg", ".jpeg"))
]

BASE_URL = "http://127.0.0.1:8080/filter"
COUNTER = 0
CONNECTER = aiohttp.TCPConnector(limit_per_host=5)

async def process_file(client: aiohttp.ClientSession, filename: str):
    global COUNTER
    async with aiofiles.open(filename, mode="rb") as f:
        content = await f.read()
    params = {
        "image": content,
        "filter_name": "sharpen",
    }
    async with client.post(BASE_URL, data=params) as resp:
        resp = await resp.read()
    path_to_save = filename.split("/")[-1]
    async with aiofiles.open(f"results/{path_to_save}", "wb") as f:
        await f.write(resp)

    COUNTER += 1

async def main():
    async with aiohttp.ClientSession() as client:
        tasks = [process_file(client, filename) for filename in files]
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    if not os.path.exists("results"):
        os.mkdir("results")
    loop = asyncio.get_event_loop()
    t0 = time.time()
    loop.run_until_complete(main())
    print(f"Processed: {COUNTER} | Elapsed : {round(time.time() - t0, 5)}")
```

