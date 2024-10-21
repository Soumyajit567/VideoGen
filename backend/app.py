from fastapi import FastAPI
from pydantic import BaseSettings
from routes.auth_routes import router as auth_router
from routes.video_routes import router as video_router
from routes.chat_routes import router as chat_router
from database import Base, engine
from config import settings
from fastapi.middleware.cors import CORSMiddleware
from fastapi_jwt_auth import AuthJWT

app = FastAPI()

# Create the tables in the database
# Base.metadata.create_all(bind=engine)

# Register routes
app.include_router(auth_router, prefix="/api/auth", tags=["Auth"])
app.include_router(video_router, prefix="/api/video", tags=["Video"])
app.include_router(chat_router, prefix="/api/chat", tags=["Chat"])

# JWT Auth Configuration
class AuthJWTSettings(BaseSettings):
    authjwt_secret_key: str = settings.AUTHJWT_SECRET_KEY
    authjwt_header_type: str = 'Bearer'
    authjwt_access_token_expires: int = 3600  # Token expires in 1 hour
    authjwt_token_location: set = {'headers'}

@AuthJWT.load_config
def get_config():
    return AuthJWTSettings()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust to your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Save the original openapi method
app.openapi_original = app.openapi

# Custom OpenAPI schema definition to add bearer authorization globally
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    # Call the original openapi method
    openapi_schema = app.openapi_original()
    openapi_schema["components"]["securitySchemes"] = {
        "bearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT"
        }
    }
    # Apply security globally
    for path in openapi_schema["paths"].values():
        for operation in path.values():
            operation.setdefault("security", []).append({"bearerAuth": []})
    app.openapi_schema = openapi_schema
    return app.openapi_schema

# Assign the custom OpenAPI function
app.openapi = custom_openapi

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
