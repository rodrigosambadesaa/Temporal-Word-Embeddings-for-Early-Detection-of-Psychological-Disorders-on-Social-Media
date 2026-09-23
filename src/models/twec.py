from __future__ import annotations

import copy
import logging
import multiprocessing
import os
from itertools import chain

import numpy as np
from gensim import utils
from gensim.models import callbacks
from gensim.models.word2vec import LineSentence, PathLineSentences, Word2Vec
from gensim.utils import tokenize
from tqdm import tqdm


class MyCallback(callbacks.CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0
        self.loss_previous_step = 0.0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        epoch_loss = loss if self.epoch == 0 else loss - self.loss_previous_step
        logging.info("Training loss after epoch %s: %s", self.epoch, epoch_loss)
        self.epoch += 1
        self.loss_previous_step = loss


class TWEC:
    """Temporal Word Embeddings with a Compass."""

    def __init__(
        self,
        size=100,
        sg=0,
        siter=10,
        ns=10,
        window=5,
        alpha=0.025,
        min_count=5,
        workers=2,
        test="test",
        init_mode="hidden",
    ):
        if size <= 0:
            raise ValueError("size must be positive")
        if siter <= 0:
            raise ValueError("siter must be positive")
        if workers <= 0:
            raise ValueError("workers must be positive")
        if init_mode not in {"hidden", "both", "copy"}:
            raise ValueError("init_mode must be one of: hidden, both, copy")

        self.size = size
        self.sg = sg
        self.trained_slices = {}
        self.gvocab = []
        self.epoch = siter
        self.negative = ns
        self.window = window
        self.static_alpha = alpha
        self.dynamic_alpha = alpha
        self.min_count = min_count
        available_cpus = multiprocessing.cpu_count()
        self.workers = max(1, min(workers, available_cpus))
        self.test = test
        self.init_mode = init_mode
        self.compass: Word2Vec | None = None

    def initialize_from_compass(self, model: Word2Vec | None) -> Word2Vec:
        if self.compass is None:
            raise RuntimeError("Compass model is not initialized")

        if self.init_mode == "copy":
            return copy.deepcopy(self.compass)

        if model is None:
            raise RuntimeError("Slice model is not initialized")
        if self.compass.layer1_size != self.size:
            raise ValueError("Compass and slice have different vector sizes")

        if len(model.wv.index_to_key) == 0:
            model.build_vocab(corpus_iterable=[self.compass.wv.index_to_key])

        vocab_m = model.wv.index_to_key
        shared_words = [w for w in vocab_m if w in self.compass.wv.key_to_index]
        indices = [self.compass.wv.key_to_index[w] for w in shared_words]

        if len(shared_words) != len(vocab_m):
            raise ValueError("Slice vocabulary contains words not present in the compass")

        model.syn1neg = np.array([self.compass.syn1neg[index] for index in indices])

        if self.init_mode == "both":
            model.wv.vectors = np.array(
                [self.compass.wv.vectors[index] for index in indices]
            )

        model.alpha = self.dynamic_alpha
        return model

    def internal_trimming_rule(self, word, count, min_count):
        return utils.RULE_KEEP if word in self.gvocab else utils.RULE_DISCARD

    @staticmethod
    def _materialize_sentences(sentences):
        if isinstance(sentences, list):
            return sentences
        return list(sentences)

    def train_model(self, sentences) -> Word2Vec:
        sentences = self._materialize_sentences(sentences)
        if not sentences:
            raise ValueError("Cannot train TWEC on an empty corpus")

        model: Word2Vec | None = None
        if self.compass is None or self.init_mode != "copy":
            model = Word2Vec(
                sg=self.sg,
                vector_size=self.size,
                alpha=self.static_alpha,
                negative=self.negative,
                window=self.window,
                min_count=self.min_count,
                workers=self.workers,
            )
            model.build_vocab(
                corpus_iterable=sentences,
                trim_rule=self.internal_trimming_rule if self.compass is not None else None,
            )

        if self.compass is not None:
            model = self.initialize_from_compass(model)

        if model is None:
            raise RuntimeError("Unable to initialize Word2Vec model")

        total_words = sum(len(sentence) for sentence in sentences)
        if total_words == 0:
            raise ValueError("Cannot train TWEC on a corpus without tokens")

        callbacks_list = [MyCallback()] if self.compass is None else []
        model.train(
            corpus_iterable=sentences,
            total_words=total_words,
            epochs=self.epoch,
            compute_loss=True,
            callbacks=callbacks_list,
        )
        return model

    def train_compass(self, chunks):
        texts = list(chain.from_iterable(chunks))
        sentences = [
            list(tokenize(str(text), lowercase=True, deacc=True))
            for text in tqdm(texts, desc="Preparing full corpus")
        ]
        logging.info("Training TWEC compass")
        self.compass = self.train_model(sentences)
        self.gvocab = list(self.compass.wv.index_to_key)

    def train_slice(self, chunks) -> Word2Vec:
        if self.compass is None:
            raise RuntimeError("Compass model is not initialized")

        sentences = [
            list(tokenize(str(text), lowercase=True, deacc=True)) for text in chunks
        ]
        return self.train_model(sentences)

    def finetune_model(self, sentences, pretrained_path):
        sentences = self._materialize_sentences(sentences)
        model: Word2Vec | None = None

        if self.compass is None or self.init_mode != "copy":
            model = Word2Vec(
                sg=self.sg,
                vector_size=self.size,
                alpha=self.static_alpha,
                negative=self.negative,
                window=self.window,
                min_count=self.min_count,
                workers=self.workers,
            )
            model.build_vocab(
                corpus_iterable=sentences,
                trim_rule=self.internal_trimming_rule if self.compass is not None else None,
            )
            model.wv.intersect_word2vec_format(
                pretrained_path, binary=True, lockf=1.0
            )

        if self.compass is not None:
            model = self.initialize_from_compass(model)

        if model is None:
            raise RuntimeError("Unable to initialize Word2Vec model")

        model.train(
            corpus_iterable=sentences,
            total_words=sum(len(sentence) for sentence in sentences),
            epochs=self.epoch,
            compute_loss=True,
        )
        return model

    def finetune_compass(self, compass_text, pre_path, overwrite=False, save=True):
        del overwrite, save
        sentences = PathLineSentences(compass_text)
        sentences.input_files = [
            path
            for path in sentences.input_files
            if not os.path.basename(path).startswith(".")
        ]
        logging.info("Fine-tuning TWEC compass")
        self.compass = self.finetune_model(sentences, pre_path)
        self.gvocab = list(self.compass.wv.index_to_key)

    def finetune_slice(self, slice_text, pretrained):
        if self.compass is None:
            raise RuntimeError("Compass model is not initialized")
        logging.info("Fine-tuning temporal embeddings for slice %s", slice_text)
        sentences = LineSentence(slice_text)
        return self.finetune_model(sentences, pretrained)
