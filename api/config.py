from pydantic import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Image processing server."
    app_version: str = "0.1"
    admin_email: str = "aquarokk@gmail.com"


settings = Settings()
