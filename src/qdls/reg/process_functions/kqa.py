
from ..register import registers

from qdls.gql.process_utils import parse_nodes_relations_sparql

@registers.process_function.register("kqa_test_function")
def kqa_test_function():
    """Test function."""
    print(f"This is a test function in kqa .")
    return True

@registers.process_function.register
def tokenization_for_SFT(example, tokenizer, mode='train'):
    """ 处理key为prompt, response的数据集 """
    if mode == 'test':
        input_str = example['prompt']
    else:
        input_str = example['prompt'] + example['response']

    td = tokenizer(input_str, truncation=True)
    example.update(td)
    if mode != 'test':
        example['labels'] = td.input_ids
    return example

@registers.process_function.register 
def kqa_cql_tokenization_causal(example, tokenizer, mode):
    """
        对一条数据的处理: 字符串处理、分词
    """
    if mode != 'test':
        text = "Here is a question: " + example['question'] \
            + " The CQL for this question is: " + example['cypher'] + tokenizer.eos_token
        td = tokenizer(text,  padding=False)
    else:
        text = "Here is a question: " + example['question'] + " The CQL for this question is: "
        td = tokenizer(text,  padding=False)
 
    d = {
        'input_ids': td.input_ids, 'attention_mask': td.attention_mask,
        'labels' :  td.input_ids if mode != 'test' else None,
    }
    d = dict(**example, **d)
    return d  


@registers.process_function.register 
def kqa_cql_tokenization_causal_with_gold_entity(example, tokenizer, mode):
    """ 
        golden entity extract from sparql
    """
    nodes, rels = parse_nodes_relations_sparql(example['sparql'])
    gold_ent = ", ".join(nodes)
    input_str = f" Related entities are {gold_ent} ."

    if mode != 'test':
        text = "Here is a question: " + example['question'] + input_str \
            + " The CQL for this question is: " + example['cypher'] + tokenizer.eos_token
        td = tokenizer(text,  padding=False)
    else:
        text = "Here is a question: " + example['question'] + input_str + " The CQL for this question is: "
        td = tokenizer(text,  padding=False)

    
    # td = tokenizer(input_str, padding=False)
    d = {
        'input_ids': td.input_ids, 'attention_mask': td.attention_mask,
        'labels' : None if mode == 'test' else td.input_ids
    }
    d = dict(**example, **d)
    return d 


@registers.process_function.register
def kqa_cql_tokenization_instruction_causal_with_gold_entity(example, tokenizer, mode):
    """ 
        以对话指令的形式，生成Cypher
    """
 
    nodes, rels = parse_nodes_relations_sparql(example['sparql'])
    gold_ent = ", ".join(nodes)

    if mode != 'test':
        text = f"User: here is a question: {example['question']}. Related entities are {gold_ent}. Give me the Cypher query for this question.\nAssistant:" + \
                f"{example['cypher']}" + tokenizer.eos_token
        td = tokenizer(text,  padding=False)
    else:
        text = f"User: here is a question: {example['question']}. Related entities are {gold_ent}. Give me the Cypher query for this question.\nAssistant:"
        td = tokenizer(text,  padding=False)

    d = {
        'input_ids': td.input_ids, 'attention_mask': td.attention_mask,
        'labels' : None if mode == 'test' else td.input_ids
    }
    d = dict(**example, **d)
    return d 


@registers.process_function.register
def kqa_cql_two_turns_with_gold_entitiy_and_feedback(example, tokenizer, mode):
    """ 
        User: here is a question: $Q. Related entities are $E. Give me the Cypher query for this question.
        Assistant: $Gen_CQL
        User: The DBMS returns $FeedBack. Try again.
        Assistant: $Gold_CQL
    
    """
    nodes, rels = parse_nodes_relations_sparql(example['sparql'])
    gold_ent = ", ".join(nodes)
    # input_str = f" Related entities are {gold_ent} ."

    def post_process_query(text):
        """ post process gpt generated query 
            pad pad Question The CQL for this question is: query => query
        """
        prefix = "The CQL for this question is:" 
        idx = text.rindex(prefix)
        query = text[idx + len(prefix):]
        return query.strip()


    if mode != 'test':
        rejected = post_process_query(example['rejected'])
        context = f"User: here is a question: {example['question']}. Related entities are {gold_ent}. Give me the Cypher query for this question.\n" + \
                f"Assistant: {rejected}\nUser: The DBMS returns {example['feedback']}. Try again.\n"
        text = context + f"Assistant: {example['cypher']}" + tokenizer.eos_token
        td = tokenizer(text,  padding=False)
    else:
        text = f"User: here is a question: {example['question']}. Related entities are {gold_ent}. Give me the Cypher query for this question.\nAssistant:"
        td = tokenizer(text,  padding=False)

    
    # td = tokenizer(input_str, padding=False)
    d = {
        'input_ids': td.input_ids, 'attention_mask': td.attention_mask,
        'labels' : None if mode == 'test' else td.input_ids
    }
    d = dict(**example, **d)
    return d 

@registers.process_function.register
def kqa_sparql_tokenization_instruction_causal_with_gold_entity(example, tokenizer, mode):
    """ 
        以对话指令的形式，生成sparql
    """
 
    nodes, rels = parse_nodes_relations_sparql(example['sparql'])
    gold_ent = ", ".join(nodes)
    input_str = f" Related entities are {gold_ent} ."

    if mode != 'test':
        text = f"User: here is a question: {example['question']}. Related entities are {gold_ent}. Give me the SPARQL query for this question.\nAssistant:" + \
                f"{example['sparql']}" + tokenizer.eos_token
        td = tokenizer(text,  padding=False)
    else:
        text = f"User: here is a question: {example['question']}. Related entities are {gold_ent}. Give me the SPARQL query for this question.\nAssistant:"
        td = tokenizer(text,  padding=False)

    d = {
        'input_ids': td.input_ids, 'attention_mask': td.attention_mask,
        'labels' : None if mode == 'test' else td.input_ids
    }
    d = dict(**example, **d)
    return d 


@registers.process_function.register
def kqa_sparql_two_turns_with_gold_entitiy_and_feedback(example, tokenizer, mode):
    """ 
        User: here is a question: $Q. Related entities are $E. Give me the SPARQL query for this question.
        Assistant: $Gen_SPARQL
        User: The DBMS returns $FeedBack. Try again.
        Assistant: $Gold_SPARQL
    
    """
    nodes, rels = parse_nodes_relations_sparql(example['sparql'])
    gold_ent = ", ".join(nodes)
    # input_str = f" Related entities are {gold_ent} ."

    def post_process_query(text):
        """ post process gpt generated query 
            pad pad Question The SPARQL for this question is: query => query
        """
        prefix = "The SPARQL for this question is:" 
        idx = text.rindex(prefix)
        query = text[idx + len(prefix):]
        return query.strip()


    if mode != 'test':
        rejected = post_process_query(example['rejected'])
        context = f"User: here is a question: {example['question']}. Related entities are {gold_ent}. Give me the SPARQL query for this question.\n" + \
                f"Assistant: {rejected}\nUser: The DBMS returns {example['feedback']}. Try again.\n"
        text = context + f"Assistant: {example['sparql']}" + tokenizer.eos_token
        td = tokenizer(text,  padding=False)
    else:
        text = f"User: here is a question: {example['question']}. Related entities are {gold_ent}. Give me the SPARQL query for this question.\nAssistant:"
        td = tokenizer(text,  padding=False)

    d = {
        'input_ids': td.input_ids, 'attention_mask': td.attention_mask,
        'labels' : None if mode == 'test' else td.input_ids
    }
    d = dict(**example, **d)
    return d 


