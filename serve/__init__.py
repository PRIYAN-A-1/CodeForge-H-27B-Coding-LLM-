"""CodeForge AI Serving and Autonomous Execution Pipeline."""

def get_pipeline():
    from serve.pipeline import pipeline
    return pipeline

__all__ = ["get_pipeline"]
