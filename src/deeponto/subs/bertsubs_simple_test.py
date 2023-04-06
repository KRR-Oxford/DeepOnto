from yacs.config import CfgNode
from deeponto.subs.bertsubs import BERTSubsIntraPipeline, DEFAULT_CONFIG_FILE
from deeponto.utils import FileUtils
from deeponto.onto import Ontology

'''
    The following segment of codes is for testing BERTSubs Intra-ontology subsumption, 
    with a given ontology (and training/valid subsumptions optionally), and a testing file
'''
config = CfgNode(FileUtils.load_file(DEFAULT_CONFIG_FILE))
config.onto_file = './foodon.owl'
config.train_subsumption_file = './train_subsumptions.csv'
config.valid_subsumption_file = './valid_subsumptions.csv'
config.test_subsumption_file = './test_subsumptions.csv'
config.subsumption_type = 'named_class'  # named_class, restriction
config.prompt.prompt_type = 'path'  # isolated, traversal, path

onto = Ontology(owl_path=config.onto_file)
pipeline = BERTSubsIntraPipeline(onto=onto, config=config)


'''
    The following segment of codes is for testing BERTSubs Inter-ontology subsumption (mappings), 
    with a given ontology (and training/valid subsumptions optionally), and a testing file
'''
