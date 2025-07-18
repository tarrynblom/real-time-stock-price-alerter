from pydantic import BaseModel


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    reload: bool = False
    workers: int = 1

    # Logging
    log_level: str = "info"
    access_log: bool = True

    # Security
    cors_origins: list = ["*"]

    class Config:
        env_prefix = "API_"
