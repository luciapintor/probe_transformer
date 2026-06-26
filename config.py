csv_path = "data_dataset/dataset_merged_probes_csv/data_with_label/all_A_full.csv"
model_path = "data_models/probe_encoder.pt"
baseline_path = "data_outputs/baseline_clusters.csv"
embeddings_path = "data_outputs/clusters.csv"

excluded_features = [
    "label",
    "is_ios",
    "timestamp",
    "mac",
    "ie0"
]