"""
BERTStatic class for handling BERT embeddings and pre-trained/fine-tuned BERT models in eval mode

"static" here means no gradient shift happened...
"""
import itertools
from typing import Dict, List, Optional

import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer


class BERTStaticSubs:
    def __init__(
        self,
        bert_checkpoint: str = "Bio_ClinicalBERT",
        tokenizer_path: Optional[str] = None,
        with_classifier: bool = False,
        max_length: int = 128,
    ):
        # if tokenizer path is not specified, use the same one as input BERT checkpoint
        if not tokenizer_path:
            tokenizer_path = bert_checkpoint
        if not with_classifier:
            self.model = AutoModel.from_pretrained(bert_checkpoint, output_hidden_states=True)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                bert_checkpoint, output_hidden_states=True
            )
        self.model.eval()  # set it to evaluation mode
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.max_length = max_length

    def word_embeds(self, sents: List[str], neg_layer_num: int = -1):
        """neg_layer_num: negative number of layer, e.g. -1 means the last layer,
        the default strategy is to take the embeddings from the last layer
        """
        # dict with keys 'input_ids', 'token_type_ids' and 'attention_mask'
        inputs = self.tokenizer(
            sents, padding=True, return_tensors="pt", max_length=self.max_length, truncation=True
        )
        # dict with keys 'last_hidden_state', 'pooler_output' and 'hidden_states'
        mask = inputs["attention_mask"]  # (batch_size, max_sent_len)
        with torch.no_grad():
            outputs = self.model(**inputs)  # **inputs will give values whenever the keys are called
            batch_embeds = torch.stack(
                outputs["hidden_states"], dim=0
            )  # (#layer, batch_size, max_sent_len, hid_dim)
            # embeddings taken from an *exact* layer
            batch_embeds = batch_embeds[neg_layer_num]  # (batch_size, max_sent_len, hid_dim)
            batch_embeds, mask = torch.broadcast_tensors(
                batch_embeds, mask.unsqueeze(2)
            )  # broadcast the mask tensor to (batch_size, max_sent_len, hid_dim)
            batch_embeds = torch.mul(
                batch_embeds, mask
            )  # the masked positions are zeros nowl torch.mul is point-wise multiplication
        return batch_embeds, mask

    def sent_embeds_mean(self, sents: List[str], neg_layer_num: int = -1):
        """Take the mean word embedding of specified layer as the sentence embedding"""
        batch_word_embeds, mask = self.word_embeds(
            sents, neg_layer_num=neg_layer_num
        )  # (batch_size, sent_len, hid_dim)
        sum_embeds = torch.sum(batch_word_embeds, dim=1)  # (batch_size, hid_dim)
        norm = (
            torch.sum(mask, dim=1).double().pow(-1)
        )  # (batch_size, hid_dim), storing the inverse of tokenized sentence length
        return torch.mul(sum_embeds, norm)

    def sent_embeds_cls(self, sents: List[str], neg_layer_num: int = -1):
        """Take the [cls] token embedding of specified layer as the sentence embedding"""
        batch_word_embeds, _ = self.word_embeds(
            sents, neg_layer_num=neg_layer_num
        )  # (batch_size, sent_len, hid_dim)
        return batch_word_embeds[:, 0, :]  # (batch_size, hid_dim)

    def ontotext_embeds(
        self, classtexts_batch: Dict[str, Dict], strategy: str = "mean", neg_layer_num: int = -1
    ):
        """Assume each text sentence describes the class entity independently and equally,
        then it is intuitive to take Mean(sent_embeds(text_sentences)) as its embedding;
        This method deals with batches of classtexts.
        """
        assert strategy == "mean" or strategy == "cls"

        # process the batch of {class-iri: {prop: texts}} (see OntoText class)
        classes_batch = []
        texts_batch = []
        lens_batch = []
        for class_iri, text_dict in classtexts_batch.items():
            classes_batch.append(class_iri)
            classtexts = list(itertools.chain.from_iterable(list(text_dict.values())))
            texts_batch += classtexts
            lens_batch.append(len(classtexts))

        # compute emebdding for each class entity in the batch
        sent_embeds_method = getattr(self, f"sent_embeds_{strategy}")
        texts_embeds = sent_embeds_method(texts_batch, neg_layer_num)
        class_embeds = []
        end = 0
        for i in range(len(lens_batch)):
            start = end
            end = lens_batch[i] + start
            class_texts_embeds = texts_embeds[start:end]
            assert len(class_texts_embeds) == lens_batch[i]
            class_embed = torch.mean(class_texts_embeds, dim=0)
            class_embeds.append(class_embed)

        return torch.stack(class_embeds)
