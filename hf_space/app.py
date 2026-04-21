"""
app.py - NumberSort FastAPI server for HF Spaces.

Pass CLASS to create_app (not instance) so each WebSocket gets
its own isolated NumberSortEnvironment.

Port 7860 required by HF Spaces.
"""

try:
    from openenv.core.env_server import create_app
    from models import NumberSortAction, NumberSortObservation
    from number_sort_environment import NumberSortEnvironment

    app = create_app(
        NumberSortEnvironment,   # CLASS -- not NumberSortEnvironment()
        NumberSortAction,
        NumberSortObservation,
        env_name="number_sort",
    )

except Exception as e:
    from fastapi import FastAPI
    app = FastAPI(title="NumberSort (fallback)")

    @app.get("/health")
    def health():
        return {"status": "healthy", "error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, workers=1)