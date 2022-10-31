# coding=utf-8
# Configuration for LongLuke, MSc research IST LISBOA José Malaquias
""" LongLUKE - LONG FIGER ET configuration"""

from transformers import PretrainedConfig
#from ...utils import logging


#logger = logging.get_logger(__name__)

LUKE_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "studio-ousia/luke-base": "https://huggingface.co/studio-ousia/luke-base/resolve/main/config.json",
    "studio-ousia/luke-large": "https://huggingface.co/studio-ousia/luke-large/resolve/main/config.json",
    "studio-ousia/luke-large-finetuned-open-entity": "https://huggingface.co/studio-ousia/luke-large-finetuned-open-entity/blob/main/config.json"
}


class LukeConfig_LongFIGER(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LukeModel`]. It is used to instantiate a LUKE
    model according to the specified arguments, defining the model architecture.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the LUKE model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`LukeModel`].
        entity_vocab_size (`int`, *optional*, defaults to 500000):
            Entity vocabulary size of the LUKE model. Defines the number of different entities that can be represented
            by the `entity_ids` passed when calling [`LukeModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        entity_emb_size (`int`, *optional*, defaults to 256):
            The number of dimensions of the entity embedding.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`LukeModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        use_entity_aware_attention (`bool`, defaults to `True`):
            Whether or not the model should use the entity-aware self-attention mechanism proposed in [LUKE: Deep
            Contextualized Entity Representations with Entity-aware Self-attention (Yamada et
            al.)](https://arxiv.org/abs/2010.01057).
    Examples:
    ```python
    #>>> # Initializing a LUKE configuration
    #>>> configuration = MyConfig()
    #>>> # Initializing a model from the configuration
    #>>> model = LukeModel(configuration)
    #>>> # Accessing the model configuration
    #>>> configuration = model.config
    ```"""
    model_type = "luke"

    def __init__(
        self,
        vocab_size=50267,
        entity_vocab_size=500000,
        hidden_size=1024,
        entity_emb_size=256,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=4098,
        type_vocab_size=1,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        use_entity_aware_attention=True,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        num_labels = 113,
        id2label= {
          0: "/art",
          1: "/art/film",
          2: "/astral_body",
          3: "/award",
          4: "/biology",
          5: "/body_part",
          6: "/broadcast",
          7: "/broadcast/tv_channel",
          8: "/broadcast_network",
          9: "/broadcast_program",
          10: "/building",
          11: "/building/airport",
          12: "/building/hospital",
          13: "/building/library",
          14: "/building/restaurant",
          15: "/building/sports_facility",
          16: "/building/theater",
          17: "/chemistry",
          18: "/computer",
          19: "/computer/programming_language",
          20: "/disease",
          21: "/education",
          22: "/education/department",
          23: "/education/educational_degree",
          24: "/event",
          25: "/event/attack",
          26: "/event/election",
          27: "/event/military_conflict",
          28: "/event/natural_disaster",
          29: "/event/protest",
          30: "/event/sports_event",
          31: "/finance",
          32: "/finance/currency",
          33: "/food",
          34: "/game",
          35: "/geography",
          36: "/geography/island",
          37: "/geography/mountain",
          38: "/god",
          39: "/government",
          40: "/government/government",
          41: "/government/political_party",
          42: "/government_agency",
          43: "/internet",
          44: "/internet/website",
          45: "/language",
          46: "/law",
          47: "/living_thing",
          48: "/livingthing",
          49: "/livingthing/animal",
          50: "/location",
          51: "/location/body_of_water",
          52: "/location/bridge",
          53: "/location/cemetery",
          54: "/location/city",
          55: "/location/country",
          56: "/location/county",
          57: "/location/province",
          58: "/medicine",
          59: "/medicine/drug",
          60: "/medicine/medical_treatment",
          61: "/medicine/symptom",
          62: "/metropolitan_transit",
          63: "/metropolitan_transit/transit_line",
          64: "/military",
          65: "/music",
          66: "/news_agency",
          67: "/organization",
          68: "/organization/airline",
          69: "/organization/company",
          70: "/organization/educational_institution",
          71: "/organization/fraternity_sorority",
          72: "/organization/sports_league",
          73: "/organization/sports_team",
          74: "/park",
          75: "/people",
          76: "/people/ethnicity",
          77: "/person",
          78: "/person/actor",
          79: "/person/architect",
          80: "/person/artist",
          81: "/person/athlete",
          82: "/person/author",
          83: "/person/coach",
          84: "/person/director",
          85: "/person/doctor",
          86: "/person/engineer",
          87: "/person/monarch",
          88: "/person/musician",
          89: "/person/politician",
          90: "/person/religious_leader",
          91: "/person/soldier",
          92: "/play",
          93: "/product",
          94: "/product/airplane",
          95: "/product/car",
          96: "/product/computer",
          97: "/product/instrument",
          98: "/product/ship",
          99: "/product/spacecraft",
          100: "/product/weapon",
          101: "/rail",
          102: "/rail/railway",
          103: "/religion",
          104: "/religion/religion",
          105: "/software",
          106: "/time",
          107: "/title",
          108: "/train",
          109: "/transit",
          110: "/transportation",
          111: "/transportation/road",
          112: "/written_work" },
        label2id =  {
          "/art": 0,
          "/art/film": 1,
          "/astral_body": 2,
          "/award": 3,
          "/biology": 4,
          "/body_part": 5,
          "/broadcast": 6,
          "/broadcast/tv_channel": 7,
          "/broadcast_network": 8,
          "/broadcast_program": 9,
          "/building": 10,
          "/building/airport": 11,
          "/building/hospital": 12,
          "/building/library": 13,
          "/building/restaurant": 14,
          "/building/sports_facility": 15,
          "/building/theater": 16,
          "/chemistry": 17,
          "/computer": 18,
          "/computer/programming_language": 19,
          "/disease": 20,
          "/education": 21,
          "/education/department": 22,
          "/education/educational_degree": 23,
          "/event": 24,
          "/event/attack": 25,
          "/event/election": 26,
          "/event/military_conflict": 27,
          "/event/natural_disaster": 28,
          "/event/protest": 29,
          "/event/sports_event": 30,
          "/finance": 31,
          "/finance/currency": 32,
          "/food": 33,
          "/game": 34,
          "/geography": 35,
          "/geography/island": 36,
          "/geography/mountain": 37,
          "/god": 38,
          "/government": 39,
          "/government/government": 40,
          "/government/political_party": 41,
          "/government_agency": 42,
          "/internet": 43,
          "/internet/website": 44,
          "/language": 45,
          "/law": 46,
          "/living_thing": 47,
          "/livingthing": 48,
          "/livingthing/animal": 49,
          "/location": 50,
          "/location/body_of_water": 51,
          "/location/bridge": 52,
          "/location/cemetery": 53,
          "/location/city": 54,
          "/location/country": 55,
          "/location/county": 56,
          "/location/province": 57,
          "/medicine": 58,
          "/medicine/drug": 59,
          "/medicine/medical_treatment": 60,
          "/medicine/symptom": 61,
          "/metropolitan_transit": 62,
          "/metropolitan_transit/transit_line": 63,
          "/military": 64,
          "/music": 65,
          "/news_agency": 66,
          "/organization": 67,
          "/organization/airline": 68,
          "/organization/company": 69,
          "/organization/educational_institution": 70,
          "/organization/fraternity_sorority": 71,
          "/organization/sports_league": 72,
          "/organization/sports_team": 73,
          "/park": 74,
          "/people": 75,
          "/people/ethnicity": 76,
          "/person": 77,
          "/person/actor": 78,
          "/person/architect": 79,
          "/person/artist": 80,
          "/person/athlete": 81,
          "/person/author": 82,
          "/person/coach": 83,
          "/person/director": 84,
          "/person/doctor": 85,
          "/person/engineer": 86,
          "/person/monarch": 87,
          "/person/musician": 88,
          "/person/politician": 89,
          "/person/religious_leader": 90,
          "/person/soldier": 91,
          "/play": 92,
          "/product": 93,
          "/product/airplane": 94,
          "/product/car": 95,
          "/product/computer": 96,
          "/product/instrument": 97,
          "/product/ship": 98,
          "/product/spacecraft": 99,
          "/product/weapon": 100,
          "/rail": 101,
          "/rail/railway": 102,
          "/religion": 103,
          "/religion/religion": 104,
          "/software": 105,
          "/time": 106,
          "/title": 107,
          "/train": 108,
          "/transit": 109,
          "/transportation": 110,
          "/transportation/road": 111,
          "/written_work": 112 },
        attention_window=[512]*24,
        classifier_dropout= None,
        position_embedding_type= "absolute",
        sep_token_id= 2,
        use_cache= True,
        **kwargs
    ):
        """Constructs LukeConfig."""
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.entity_vocab_size = entity_vocab_size
        self.hidden_size = hidden_size
        self.entity_emb_size = entity_emb_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_entity_aware_attention = use_entity_aware_attention
        self.attention_window = attention_window
        self.classifier_dropout = classifier_dropout
        self.position_embedding_type = position_embedding_type
        self.sep_token_id = sep_token_id
        self.use_cache = use_cache
        self.num_labels = num_labels
        self.id2label = id2label
        self.label2id = label2id

