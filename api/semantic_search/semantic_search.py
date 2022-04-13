import gc

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

import ownphotos

dir_clip_ViT_B_32_model = ownphotos.settings.CLIP_ROOT


class SemanticSearch:
    model_is_loaded = False

    def load(self):
        self.load_model()
        self.model_is_loaded = True

    def unload(self):
        self.model = None
        gc.collect()
        self.model_is_loaded = False

    def load_model(self):
        self.model = SentenceTransformer(dir_clip_ViT_B_32_model)

    def calculate_clip_embeddings(self, img_paths):
        if not self.model_is_loaded:
            self.load()

        if type(img_paths) is list:
            imgs = list(map(Image.open, img_paths))
        else:
            imgs = [Image.open(img_paths)]

        imgs_emb = self.model.encode(imgs, batch_size=32, convert_to_tensor=True)

        if type(img_paths) is list:
            magnitudes = map(np.linalg.norm, imgs_emb)

            return imgs_emb, magnitudes
        else:
            img_emb = imgs_emb[0].tolist()
            magnitude = np.linalg.norm(img_emb)

            return img_emb, magnitude

    def calculate_query_embeddings(self, query):
        if not self.model_is_loaded:
            self.load()

        query_emb = self.model.encode([query], convert_to_tensor=True)[0].tolist()
        magnitude = np.linalg.norm(query_emb)

        return query_emb, magnitude


semantic_search_instance = SemanticSearch()
