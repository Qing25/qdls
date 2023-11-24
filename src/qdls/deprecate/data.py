


def is_sample_valid(sample):
    """ span 抽取问题（SQuAD格式）,数据标注是否正确
    Args:
        sample (SquadSample): 读入的单条样本对象
    Returns:
        bool: 该数据的标注是否正确
    """
    if sample.answer_texts != []:
        text = sample.answer_texts[0]
        start_char = sample.answer_starts[0]
        # 标注的start_char的位置有问题
        if not sample.context[start_char:start_char+len(text)] == text:
            return False
    else:
        if sample.answer_starts != []:
            return False
    return True