import fastapi
from fastapi.exceptions import HTTPException, RequestValidationError
from fastapi.responses import JSONResponse
from challenge.model import DelayModel
from challenge.schemas import PredictRequest, PredictResponse
import pandas as pd

app = fastapi.FastAPI()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request, exc):
    """Convert Pydantic validation errors to HTTP 400 responses."""
    # Extract first error message
    errors = exc.errors()
    if errors:
        error_message = errors[0]["msg"]
        detail = f"Invalid {errors[0]['loc'][0]}: {error_message}"
    else:
        detail = "Invalid request"
    return JSONResponse(status_code=400, content={"detail": detail})


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}


@app.post("/predict", status_code=200)
async def post_predict(payload: PredictRequest) -> PredictResponse:
    """
    Predict flight delays.

    Args:
        payload: PredictRequest containing a list of flights to predict.

    Returns:
        PredictResponse with prediction results.

    Raises:
        HTTPException: If prediction fails (500 error).
    """
    rows = []

    for flight in payload.flights:
        rows.append(
            {
                "OPERA": flight.OPERA,
                "TIPOVUELO": flight.TIPOVUELO,
                "MES": flight.MES,
            }
        )

    try:
        df = pd.DataFrame(rows)
        model = DelayModel()

        features = model.preprocess(df)

        predictions = model.predict(features)
        return PredictResponse(predict=predictions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
