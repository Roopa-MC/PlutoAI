# -------------------------------------------------
# Build Embeddings Utility
#
# This program is executed only when the knowledge
# base changes.
#
# It generates embeddings for every document in the
# knowledge folder and stores them in embeddings.json
# -------------------------------------------------

from embeddings import create_embeddings

create_embeddings()
