# -*- coding: utf-8 -*-
# @File    :   utils.py
# @Time    :   2023/03/13 14:40:16
# @Author  :   Qing 
# @Email   :   aqsz2526@outlook.com
######################### docstring ########################
'''

'''

from antlr4.tree.Tree import TerminalNodeImpl
from .cypher.utils.parse import parse_cypher
from .sparql.utils.parse import parse_sparql
from .utils import parse_layer

_CY_NODES = [
    # 'oC_SymbolicName',              # pv, e_1
    'oC_RelationshipTypes',         # :for_work :instance_of
    # 'oC_NodeLabel',                 # : Resource :Relation
    # 'oC_RelationshipDetail',            # [:instance_of]
    # 'oC_PropertyKeyName',           # value
    # 'oC_NumberLiteral',
    # 'oC_MapLiteral',                   # {name:"xxx"}
    # 'oC_Expression', 
    'oC_Literal'                    # "a string or number"
]

_SP_NODES = [
    # 'var',                          # ?v ?pv
    # 'pathPrimary',                  # <pred:value> date name fact_r h t 
    'iri',                          #  <for_work>
    'string',                       # "a string"
    'numericLiteralUnsigned',       # 1895
    'numericLiteralNegative'        # -192
]

def parse_subtree(tree, parts):
    if tree.getText() == "<EOF>":
        return
    elif isinstance(tree, TerminalNodeImpl) or tree.children is None:
        parts.append(tree.getText())
    else:
        for child in tree.children:
            parse_subtree(child, parts)

def get_subtree_leaves(tree):
    """ 遍历一个子树、返回其叶子结点的list """
    parts = [] 
    parse_subtree(tree, parts)
    return parts

# classnames = load_json("/home/qing/workspace/papers/query_skeleton/src/data_utils/class_names.json")
classnames = []

def parse_skeleton(tree, parts, ruleNames, idx2value, lang='sparql'):
    """
        将 ast 解析为骨架，保留 AST2Skeleton._SP_NODES 中定义的非终结符
        parts: 骨架节点
        idx2value : 非终结符到骨架节点idx 到 其叶子结点的映射
    """
    if tree.getText() == "<EOF>":
        return
    elif isinstance(tree, TerminalNodeImpl):
        parts.append(tree.getText())
    elif tree.children is None:
        return 
    else:
        nonterm_name = ruleNames[tree.getRuleIndex()]
        # print(lang, nonterm_name, AST2Skeleton.nonterms(lang))
        if nonterm_name in _CY_NODES + _SP_NODES:  # 如果是预定义的 非终结符
            values = get_subtree_leaves(tree)
            # print(len(parts)-1, nonterm_name, values)
            # if values[0][1:-1] in ["human", "percentage", "film", "city", "square metre", "square kilometre", "county", "metre"]:
            if values[0][1:-1] in classnames:
                pass  # instance of or unit 不算需要替换的
            else:
                parts.append(nonterm_name)
                idx2value[len(parts)-1] = (nonterm_name, values)
        else:
            for child in tree.children:
                parse_skeleton(child, parts, ruleNames, idx2value, lang=lang)


def parse_nodes_relations_cql(cy):
    """
    从Cypher中解析出节点和关系
    nodes:      ['{name:"human"}', '{value:"TheOliverStone"}', '{name:"Nixon"}', '{name:"Twitter_username"}']
    relations:  ['[:instance_of]', '[:Twitter_username]', '[:director]', '[:fact_h]', '[:fact_t]', '[:fact_r]', '[:number_of_subscribers]']
    """
    tree, parser = parse_cypher(cy,True)
    parts, d = [], {} 
    parse_skeleton(tree, parts, parser.ruleNames, d, lang='cypher')
    # print(parts)
    # print(d)
    nodes, relations = [], []
    for k,v in d.items():
        nonterm_name, values = v 
        if nonterm_name == "oC_RelationshipDetail" or nonterm_name == "oC_RelationshipTypes":
            relations.append( "".join(values) )
        # elif nonterm_name == "oC_MapLiteral":
        # elif nonterm_name == "oC_Expression":
            # nodes.append( "".join(values) )
        elif nonterm_name == "oC_Literal":
            nodes.append( "".join(values) )
        else:
            raise Exception(f'{nonterm_name} not handled!')
    
    # print(nodes)
    # print(relations)
    return nodes, relations 

def parse_nodes_relations_sparql(sp):
    """ 
        从sparql中解析出节点和关系
        nodes:      ['"human"', '"TheOliverStone"', '"Nixon"', '"Twitter_username"']
        relations:  ['<instance_of>', '<Twitter_username>', '<director>', '<fact_h>', '<fact_t>', '<fact_r>', '<number_of_subscribers>']
    """
    tree, parser = parse_sparql(sp,True)
    parts, d = [], {} 
    parse_skeleton(tree, parts, parser.ruleNames, d)
    # print(parts)
    # print(d)
    nodes, relations = [], []
    for k,v in d.items():
        nonterm_name, values = v 
        if nonterm_name in ["numericLiteralUnsigned", "numericLiteralNegative", "string"]:
            nodes.append( "".join(values) )
        elif nonterm_name in ['iri']:
            relations.append( "".join(values) )
        else:
            raise Exception(f'{nonterm_name} not handled!')
    
    # print(nodes)
    # print(relations)
    return nodes, relations 


def tokenize_sequence(tokenizer, list_of_str):
    """ 
        bartTokenizer for multi str
    """
    input_ids = [tokenizer.bos_token_id]
    for s in list_of_str:
        ids = tokenizer.encode(s)
        input_ids.extend(ids[1:]+ [tokenizer.eos_token_id])
        
    input_ids = input_ids[:-1]
    attention_mask = [1 for _ in input_ids]

    return input_ids, attention_mask



def tokenize_with_type_ids(tokenizer, input_ids, gql, offsets, lang='cypher'):
    """ 
        0 for normal, 1 for node , 2 for relation
    """
    node_target = [0 for _ in input_ids]
    rel_target = [0 for _ in input_ids]

    if lang == 'cypher':
        nodes, relations = parse_nodes_relations_cql(gql)
    elif lang == 'sparql':
        nodes, relations = parse_nodes_relations_sparql(gql)
    else:   
        raise Exception(f"{lang} not supported!")

    # for x in nodes:
    #     flag = False
    #     s = tokenizer(x).input_ids[1:-1]
    #     l = len(s)
    #     for i in range(1, len(input_ids)-l):
    #         if input_ids[i:i+l] == s:
    #             rel_target[i:i+l] = [1 for _ in s]
    #             flag = True
    #     if flag == False:
    #         raise Exception(f"No matched: {x} in {gql}")
    
    for n in nodes:
        s = gql.index(n)
        e = s+len(n)
        # print(s,e, gql[s:e])
        ts, te = None, None
        for i,(_s,_e) in enumerate(offsets):
            if _s == s or (_s < s <= _e):
                ts = i
            if _e == e or (_s <= e < _e):
                te = i 
        if ts is None or te is None:
            raise Exception(f"No matched: {n} in {gql}\n{(s,e)} | {(ts, te)} not in {offsets}")
        else:
            node_target[ts:te+1] = [1 for _ in range(ts,te+1)]

    for n in relations:
        s = gql.index(n)
        e = s+len(n)
        # print(s,e, gql[s:e])
        ts, te = None, None
        for i,(_s,_e) in enumerate(offsets):
            if _s == s or (_s < s <= _e):
                ts = i
            if _e == e or (_s <= e < _e):
                te = i 
        if ts is None or te is None:
            raise Exception(f"No matched: {n} in {gql}\n{(s,e)} | {(ts, te)} not in {offsets}")
        else:
            rel_target[ts:te+1] = [1 for _ in range(ts,te+1)]
            

    merged = [ 1 if n or r else 0 for n,r in zip(node_target, rel_target) ]

    return node_target, rel_target, merged
