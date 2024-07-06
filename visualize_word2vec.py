from tensorboard.plugins.projector import visualize_embeddings

from tensorboard.plugins.projector import ProjectorConfig

config = ProjectorConfig()

embeddings = "../DS_lab/new_biobert_emb_terms_data.tsv"
metadata = "../DS_lab/new_biobert_emb_terms_data_md.tsv"

visualize_embeddings(config, embeddings)