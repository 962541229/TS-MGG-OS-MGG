from build_word_word_Topic_graph import construct_topic_graph
from build_word_word_PMI_graph import construct_pmi_graph
from build_word_word_Dis_graph import construct_dis_graph

# from Dataset.ng20.config import Config
from Dataset.ng20.config import Config
# from Dataset.ohsumed.config import Config
# from Dataset.R8.config import Config
# from Dataset.R52.config import Config

config = Config()

construct_topic_graph(config)
construct_pmi_graph(config)
construct_dis_graph(config)
