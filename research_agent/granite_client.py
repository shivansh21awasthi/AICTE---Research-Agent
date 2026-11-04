"""
granite_client.py — Thin wrapper around IBM watsonx.ai / Granite.

Provides:
  - GraniteClient.generate()   → text generation (summarise, draft, hypothesise)
  - GraniteClient.embed()      → text embeddings (semantic similarity)
"""

from __future__ import annotations

import logging
from typing import Optional

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference, Embeddings
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

from .config import config

logger = logging.getLogger("research_agent.granite")


class GraniteClient:
    """
    Wraps IBM watsonx.ai ModelInference for text generation and embeddings
    using IBM Granite models on the IBM Cloud Lite tier.
    """

    def __init__(self) -> None:
        self._credentials = Credentials(
            url=config.watsonx_url,
            api_key=config.ibm_cloud_api_key,
        )
        self._project_id = config.watsonx_project_id

        # Lazy-loaded model instances
        self._gen_model: Optional[ModelInference] = None
        self._emb_model: Optional[Embeddings] = None

    # ── Private helpers ──────────────────────────────────────────────────────

    def _get_gen_model(self) -> ModelInference:
        if self._gen_model is None:
            logger.debug("Initialising Granite generation model: %s", config.granite_model_id)
            self._gen_model = ModelInference(
                model_id=config.granite_model_id,
                credentials=self._credentials,
                project_id=self._project_id,
                params={
                    GenParams.MAX_NEW_TOKENS: config.max_new_tokens,
                    GenParams.TEMPERATURE: config.temperature,
                    GenParams.TOP_P: config.top_p,
                    GenParams.STOP_SEQUENCES: ["<|endoftext|>"],
                },
            )
        return self._gen_model

    def _get_emb_model(self) -> Embeddings:
        if self._emb_model is None:
            logger.debug("Initialising Granite embedding model: %s", config.granite_embedding_model_id)
            self._emb_model = Embeddings(
                model_id=config.granite_embedding_model_id,
                credentials=self._credentials,
                project_id=self._project_id,
            )
        return self._emb_model

    # ── Public API ───────────────────────────────────────────────────────────

    def generate(self, prompt: str) -> str:
        """
        Send a prompt to IBM Granite and return the generated text.

        Parameters
        ----------
        prompt : str
            The full instruction + context prompt.

        Returns
        -------
        str
            Generated text from IBM Granite.
        """
        logger.debug("Sending generation request (%d chars)", len(prompt))
        try:
            model = self._get_gen_model()
            response = model.generate_text(prompt=prompt)
            return response.strip() if isinstance(response, str) else str(response)
        except Exception as exc:
            logger.error("Granite generation failed: %s", exc)
            raise

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Return embedding vectors for a list of texts using Granite embeddings.

        Parameters
        ----------
        texts : list[str]
            Texts to embed.

        Returns
        -------
        list[list[float]]
            One embedding vector per input text.
        """
        logger.debug("Embedding %d texts", len(texts))
        try:
            model = self._get_emb_model()
            result = model.embed_documents(texts=texts)
            return result
        except Exception as exc:
            logger.error("Granite embedding failed: %s", exc)
            raise

    def summarize(self, text: str, max_sentences: int = 5) -> str:
        """Convenience wrapper: summarise a block of text."""
        prompt = (
            f"You are an expert research assistant. "
            f"Summarise the following academic text in at most {max_sentences} clear sentences. "
            f"Focus on the main contributions, methodology, and findings.\n\n"
            f"Text:\n{text}\n\n"
            f"Summary:"
        )
        return self.generate(prompt)

    def suggest_hypotheses(self, context: str, n: int = 3) -> str:
        """Ask Granite to propose research hypotheses from a given context."""
        prompt = (
            f"You are a senior research scientist. "
            f"Based on the research context below, propose {n} novel, testable research hypotheses. "
            f"Number each hypothesis clearly.\n\n"
            f"Context:\n{context}\n\n"
            f"Hypotheses:"
        )
        return self.generate(prompt)

    def draft_section(self, section_title: str, notes: str) -> str:
        """Draft a section of a research paper from bullet-point notes."""
        prompt = (
            f"You are an academic writer. "
            f"Write a well-structured paragraph for the '{section_title}' section of a research paper "
            f"based on the following notes. Use formal academic language.\n\n"
            f"Notes:\n{notes}\n\n"
            f"Section text:"
        )
        return self.generate(prompt)
