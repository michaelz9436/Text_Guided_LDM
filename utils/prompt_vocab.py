# prompt_vocab.py
"""
Text2Mol Prompt Vocabulary Library
用于管理所有生成自然语言描述符的同义词、形容词、连接词和句式模板。
"""

NOUNS = [
    "molecule", "compound", "chemical", "substance", "structure", 
    "target", "derivative", "entity", "scaffold", "agent"
]

SENTENCE_STARTERS = [
    "This is a", "It's a", "This target is a", "A novel", "A", 
    "An interesting", "The proposed structure is a", "Looks like a",
    "We can describe this as a", "Essentially, it is a"
]

# 物理与几何形容词库
ADJECTIVES = {
    "large": [
        "bulky", "large", "heavyweight", "high-molecular-weight", 
        "sterically demanding", "macromolecule-like", "massive"
    ],
    "small": [
        "small", "compact", "low-molecular-weight", "lightweight", 
        "tiny", "miniature"
    ],
    "medium": [
        "medium-sized", "mid-sized", "moderately sized"
    ],
    "lipophilic": [
        "lipophilic", "hydrophobic", "greasy", "non-polar", "fat-soluble", 
        "highly lipophilic"
    ],
    "hydrophilic": [
        "hydrophilic", "polar", "water-soluble", "highly polar", 
        "polarity-rich"
    ],
    "rigid": [
        "rigid", "conformationally stiff", "conformationally locked", 
        "inflexible", "structurally rigid", "stiff"
    ],
    "flexible": [
        "highly flexible", "floppy", "chain-like", "conformationally free",
        "structurally flexible", "loose"
    ]
}

# 环系统骨架描述库
RING_SCAFFOLDS = {
    "acyclic": [
        "acyclic structure", "linear aliphatic chain", "non-cyclic backbone", 
        "open-chain framework", "ring-free scaffold"
    ],
    "fully_aromatic": [
        "fully aromatic framework", "conjugated aromatic core", 
        "strictly aromatic backbone", "aromatic system"
    ],
    "aliphatic_cyclic": [
        "aliphatic cyclic backbone", "saturated ring system", 
        "non-aromatic cyclic core", "alicyclic scaffold"
    ],
    "mixed": [
        "mixed aromatic-aliphatic scaffold", "hybrid cyclic system", 
        "complex ring system combining saturated and aromatic parts",
        "scaffold containing both aromatic and aliphatic rings"
    ]
}

# 官能团描述前缀/后缀
HALOGEN_DESCRIPTIONS = [
    "halogen substituents", "some halogens", "halogenated sites", 
    "halogen atoms", "a degree of halogenation"
]

# 句子连接词库 (用于连接形容词/骨架 -> 官能团)
TRANSITIONS_TO_FG = [
    ", which is further decorated with ",
    ", featuring ",
    ", adorned with ",
    ", functionalized by ",
    ", bearing ",
    " that incorporates ",
    " coupled with ",
    ", and is substituted with ",
    ". It possesses ",
    ". The structure is characterized by the presence of ",
    " containing "
]

# 句子连接词库 (用于连接两个独立的短句)
TRANSITIONS_BETWEEN_SENTENCES = [
    " Additionally, it features ",
    " Furthermore, it contains ",
    " Notable functional groups include ",
    " The molecule also incorporates ",
    " It also bears "
]

# MIDJOURNEY 风格分隔符
TAG_SEPARATORS = [
    " | ", " - ", ", ", " ; "
]