import sys

from numpy import prod
from nltk.corpus import wordnet as wn


def wup_measure(a,b,similarity_threshold=0.925):
    """ Returns Wu-Palmer similarity score.
        More specifically, it computes:
        max_{x \in interp(a)} max_{y \in interp(b)} wup(x,y)
        where interp is a 'interpretation field'
    """
    def get_semantic_field(a):
        weight = 1.0
        semantic_field = wn.synsets(a,pos=wn.NOUN)
        return (semantic_field,weight)


    def get_stem_word(a):
        """
        Sometimes answer has form word\d+:wordid.
        If so we return word and downweight
        """
        weight = 1.0
        return (a,weight)


    global_weight=1.0

    (a,global_weight_a)=get_stem_word(a)
    (b,global_weight_b)=get_stem_word(b)
    global_weight = min(global_weight_a,global_weight_b)

    if a==b:
        # they are the same
        return 1.0*global_weight

    if a==[] or b==[]:
        return 0


    interp_a,weight_a = get_semantic_field(a) 
    interp_b,weight_b = get_semantic_field(b)

    if interp_a == [] or interp_b == []:
        return 0

    # we take the most optimistic interpretation
    global_max=0.0
    for x in interp_a:
        for y in interp_b:
            local_score=x.wup_similarity(y)
            if local_score > global_max:
                global_max=local_score

    # we need to use the semantic fields and therefore we downweight
    # unless the score is high which indicates both are synonyms
    if global_max < similarity_threshold:
        interp_weight = 0.1
    else:
        interp_weight = 1.0

    final_score=global_max*weight_a*weight_b*interp_weight*global_weight
    return final_score 

def wup_measure_sentences(answer_true, answer_pred, similarity_threshold=0.9):
    """
    :param answer_true: list of words from the answer
    :param answer_pred: list of words from the predicted answer
    :param similarity_threshold: WUPS similarity threshold (default = 0.9)
    :return: the average of the WUPS score for both sentences
    """
    sum = 0
    num_ans = min(len(answer_true), len(answer_pred))
    for i in range(num_ans):
        sum += wup_measure(answer_true[i], answer_pred[i], similarity_threshold)
    return  sum/len(answer_true)

def wup_measure_sequences(answer_true, answer_pred, inv_map, similarity_threshold=0.9):
    """
    :param answer_true: list of words from the answer
    :param answer_pred: list of words from the predicted answer
    :param similarity_threshold: WUPS similarity threshold (default = 0.9)
    :return: the average of the WUPS score for both sentences
    """
    sum = 0
    for i in range(min(len(answer_true), len(answer_pred))):
        sum += wup_measure(inv_map[answer_true[i]], inv_map[answer_pred[i]], similarity_threshold)
    if len(answer_true) == 0:
        return 0
    return  sum/len(answer_true)


