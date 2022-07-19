import os
import sys

main_dir = os.getcwd().split("DeepOnto")[0] + "DeepOnto/src"
sys.path.append(main_dir)

from deeponto.utils import read_jsonl
from deeponto.onto.text import Tokenizer
from deeponto import SavedObj
from rank_bm25 import BM25Okapi


import click


@click.command()
@click.option("-o", "--saved_path", type=click.Path(exists=True), default=".")
@click.option("-d", "--jsonl_data_path", type=click.Path(exists=True))
@click.option(
    "-t", "--tokenizer_type", type=click.Choice(["pre-trained", "rule-based"])
)  # sub-word or word level
@click.option(
    "-p", "--pretrained_tokenizer_path", type=str, default="bert-base-uncased"
)  # if choosing pre_trained tokenizer, load huggingface tokenizer
@click.option("-n", "--num_candidates", type=int)
def main(
    saved_path: str,
    jsonl_data_path: str,
    tokenizer_type: str,
    pretrained_tokenizer_path: str,
    num_candidates: int,
):
    jsonl_data = read_jsonl(jsonl_data_path)
    if tokenizer_type == "pre-trained":
        tkz = Tokenizer.from_pretrained(pretrained_path=pretrained_tokenizer_path)
        print(f"Load subword-level tokenizer from (huggingface): {pretrained_tokenizer_path}")
    else:
        tkz = Tokenizer.from_rule_based(spacy_lib_path="en_core_web_sm")
        print(f"Load word-level tokenizer from (spacy): en_core_web_sm")
        
    # build the tokenized corpus
    # NOTE: change the property here if using other keys
    # NOTE: lower-case everything here
    entity_label2tokens = dict()  # map the original string to tokenized string
    entity_label_corpus = []
    for entry in jsonl_data:
        label = entry["label_title"]
        entity_label2tokens[label] = tkz.tokenize(entry["label_title"].lower())
        entity_label_corpus.append(entity_label2tokens[label])
    entity_tokens2label = {" ".join(v): k for k, v in entity_label2tokens.items()}
    bm25_model = BM25Okapi(entity_label_corpus)

    cand_dict = dict()
    for entry in jsonl_data:
        mention = entry["mention"].lower()
        top_n_cands = bm25_model.get_top_n(tkz.tokenize(mention), entity_label_corpus, n=num_candidates)
        top_n_cands = [entity_tokens2label[" ".join(cand)] for cand in top_n_cands]
        cand_dict[mention] = top_n_cands

    SavedObj.save_json(cand_dict, saved_path + "/mention2candidates.json")
    
main()
