from fastapi import FastAPI
from starlette.datastructures import UploadFile as StarletteUploadFile

from api.routes.images import router
from api.utils.executor import executor

from . import config

# keep the SpooledTemporaryFile in-memory
# StarletteUploadFile.spool_max_size = 0


async def about():
    return {
        "app_name": config.settings.app_name,
        "app_version": config.settings.app_version,
        "app_status": "running",
    }


def create_app() -> FastAPI:
    app = FastAPI()

    app.add_event_handler("startup", executor.startup)
    app.add_event_handler("shutdown", executor.shutdown)

    app.include_router(router)

    app.add_api_route("/about", about)

    return app
