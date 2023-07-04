import mlflow

def entry_point(parameters):
  """Entry point for the MLflow project."""
  # Log a metric.
  mlflow.log_metric("accuracy", 0.99)
  # Log an artifact.
  mlflow.log_artifact("model.pkl")
  return

if __name__ == "__main__":
  mlflow.run(".", entry_point=entry_point)
