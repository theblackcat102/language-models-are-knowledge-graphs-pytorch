

found_invalid = [
    'and', 'of', 'in', 'to', ',', 'for', 'be', 'by', 'with', 'on', 'as', 'that', 'from', 'be', ')', '(', 'which',
    'at', 'be', 'be', 'be', ';', 'or', 'but', 'have', 'have', 'the', 'have', 'not', 'after', '"', 'include', 'also',
    'be', 'into', 'between', 'such', ':', 'do', 'while', 'when', 'during', 'would', 'over', 'since', '2019', 
    'well', 'than', '2020', 'under', 'where', 'one', 'be', 'hold', '2018', 'can', 'through', '-', 
    'make',  'out', 'there', 'know', 'due', 'a', 'take', 'up', 'begin', 'before', 'about',
    "'",  '4', '10', '3', '11', '&', '$', '12',  '2015', '2008','–', 'will',
    'so', 'do', 'follow', 'most', 'although', 'cause', 'only', '—',  '2007',  '2014', 'mostly', '5', 'say', '2017', '20', 
    '2009',
]

invalid_relations = [
    'am', 'are', 'is', 'was', 'were', 'been', 'being', # be verb
    'in', 'on', 'at', 'of', 'for', 'with', 'from',# preposition
    'and', 'but', 'or', 'so', 'because', 'when', 'before', 'although', # conjunction
    'oh', 'wow', 'ouch', 'ah', 'oops',
    'what', 'how', 'where', 'when', 'who', 'whom',
    'a', 'and', 'the', 'there', 
    'them', 'he', 'she', 'him', 'her', 'it', # pronoun
    'ten', 'hundred', 'thousand', 'million', 'billion',# unit
    'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',# number
    'has', 'have', 'been',
    'year', 'month', 'day', 'daily',
] + found_invalid


invalid_relations_set = set(invalid_relations)