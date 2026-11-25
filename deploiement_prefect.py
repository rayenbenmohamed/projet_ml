"""Déploiements Prefect avec planification quotidienne."""

from prefect import serve

from pipeline_prefect import (
    all_flow,
    code_flow,
    evaluate_flow,
    install_flow,
    predict_flow,
    train_flow,
)

if __name__ == "__main__":
    serve(
        all_flow.to_deployment(
            name="ml-pipeline-all",
            cron="0 2 * * *",
            tags=["full-pipeline", "mlops"],
        ),
        train_flow.to_deployment(
            name="ml-pipeline-train",
            tags=["training", "mlops"],
        ),
        evaluate_flow.to_deployment(
            name="ml-pipeline-evaluate",
            tags=["evaluation", "mlops"],
        ),
        predict_flow.to_deployment(
            name="ml-pipeline-predict",
            tags=["prediction", "mlops"],
        ),
        code_flow.to_deployment(
            name="ml-pipeline-code",
            tags=["code-quality", "mlops"],
        ),
        install_flow.to_deployment(
            name="ml-pipeline-install",
            tags=["setup", "mlops"],
        ),
    )
