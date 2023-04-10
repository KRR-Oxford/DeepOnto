from yacs.config import CfgNode
from deeponto.subs.bertsubs import BERTSubsIntraPipeline, DEFAULT_CONFIG_FILE_INTRA, BERTSubsInterPipeline, DEFAULT_CONFIG_FILE_INTER
from deeponto.utils import FileUtils
from deeponto.onto import Ontology

'''
    The following segment of codes is for testing BERTSubs Intra-ontology subsumption, 
    with a given ontology (and training/valid subsumptions optionally), and a testing file.
'''
config = CfgNode(FileUtils.load_file(DEFAULT_CONFIG_FILE_INTRA))
config.onto_file = './foodon.owl'
config.train_subsumption_file = './train_subsumptions.csv'
config.valid_subsumption_file = './valid_subsumptions.csv'
config.test_subsumption_file = './test_subsumptions.csv'
config.test_type = 'evaluation'
config.subsumption_type = 'named_class'  # named_class, restriction
config.prompt.prompt_type = 'isolated'  # isolated, traversal, path

onto = Ontology(owl_path=config.onto_file)
intra_pipeline = BERTSubsIntraPipeline(onto=onto, config=config)


'''
    The following segment of codes is for testing BERTSubs Inter-ontology subsumption (mappings), 
    with a given ontology (and training/valid subsumptions optionally), and a testing file
'''

config = CfgNode(FileUtils.load_file(DEFAULT_CONFIG_FILE_INTER))
config.src_onto_file = './helis2foodon/helis_v1.00.owl'
config.tgt_onto_file = './helis2foodon/foodon-merged.0.4.8.subs.owl'
config.train_subsumption_file = './helis2foodon/train_subsumptions.csv'
config.valid_subsumption_file = './helis2foodon/valid_subsumptions.csv'
config.test_subsumption_file = './helis2foodon/test_subsumptions.csv'
config.test_type = 'evaluation'
config.subsumption_type = 'named_class'  # named_class, restriction
config.prompt.prompt_type = 'path'  # isolated, traversal, path

src_onto = Ontology(owl_path=config.src_onto_file)
tgt_onto = Ontology(owl_path=config.tgt_onto_file)
inter_pipeline = BERTSubsInterPipeline(src_onto=src_onto, tgt_onto=tgt_onto, config=config)