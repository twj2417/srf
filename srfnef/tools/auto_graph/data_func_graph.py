# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: __init__.py
@date: 4/26/2019
@desc:
'''
import srfnef as nef
from graphviz import Digraph
from srfnef.tools.get_members import module_parser


def data_func_graph_genrator():
    data_dct = module_parser(nef, max_deps = 5, include = 'data_classes')['data_classes']
    func_dct = module_parser(nef, max_deps = 5, include = 'functions')['functions']

    g = Digraph('G',
                filename = srfnef.config.LOG_DIR + '/srfnef-' + nef.__version__ + '.data_func_graph.gv',
                strict = True)

    g.attr(size = '6,6')
    g.node_attr.update(color = 'lightblue2', style = 'filled')
    g.attr(rankdir = 'LR')
    with g.subgraph(name = 'basic types') as c0:
        c0.attr(rankdir = 'TB')
        c0.attr(color = 'black')
        c0.node('list')
        c0.node('str')
        c0.node('float')
        c0.node('bool')
        c0.node('tuple')
        c0.node('int')
        c0.node('object')
        c0.view()

    with g.subgraph(name = 'data types') as c1:
        c1.attr(rankdir = 'TB')
        c1.attr(color = 'blue')
        for node_name in data_dct.keys():
            c1.node(node_name)

        for node_name, node_attr in data_dct.items():
            for type_ in node_attr.values():
                if type_.startswith('List'):
                    g.edge('list', node_name, style = 'dashed')
                elif type_.startswith('Tuple'):
                    g.edge('tuple', node_name, style = 'dashed')
                elif type_ in data_dct:
                    c1.edge(type_, node_name, style = 'dashed')
                else:
                    g.edge(type_, node_name, style = 'dashed')
        c1.view()

    with g.subgraph(name = 'func types') as c2:
        c2.attr(rankdir = 'TB')
        c2.attr(color = 'red')
        for node_name in data_dct.keys():
            c2.node(node_name)

        for node_name, node_attr in func_dct.items():
            for key, type_ in node_attr.items():
                if key == '__call__':
                    continue
                if type_.startswith('List'):
                    g.edge('list', node_name, style = 'dashed')
                elif type_.startswith('Tuple'):
                    g.edge('tuple', node_name, style = 'dashed')
                elif type_ in func_dct:
                    c2.edge(type_, node_name, style = 'dashed')
                else:
                    g.edge(type_, node_name, style = 'dashed')

    g.attr(overlap = 'false')

    g.view()

    # with g.subgraph(name = 'basic types') as c0:
    #     c0.attr(style = 'filled')
    #     c0.attr(color = 'yellow')
    #     c0.node_attr._update(style = 'filled', color = 'white')
    #
    #     c0.node('list')
    #     c0.node('str')
    #     c0.node('float')
    #     c0.node('bool')
    #     c0.node('tuple')
    #     c0.node('int')
    #     c0.node('object')
    #     c0.attr(label = 'basic types')
    #
    # with g.subgraph(name = 'data_class') as c1:
    #     c1.attr(style = 'filled')
    #     c1.attr(color = 'blue')
    #     c1.node_attr._update(style = 'filled', color = 'white')
    #     for node_name in data_dct.keys():
    #         c1.node(node_name)
    #
    #     for node_name, node_attr in data_dct.items():
    #         for type_ in node_attr.values():
    #             if type_.startswith('List'):
    #                 g.edge(node_name, 'list')
    #             elif type_.startswith('Tuple'):
    #                 g.edge(node_name, 'Tuple')
    #             elif type_ not in data_dct.keys():
    #                 g.edge(node_name, type_)
    #             else:
    #                 c1.edge(node_name, type_)
    #
    #     c1.attr(label = 'data_classes')
    #
    # with g.subgraph(name = 'func_class') as c2:
    #     c1.attr(style = 'filled')
    #     c1.attr(color = 'black')
    #     c1.node_attr._update(style = 'filled', color = 'white')
    #     for node_name in func_dct.keys():
    #         c2.node(node_name)
    #
    #     for node_name, node_attr in func_dct.items():
    #
    #         for key, type_ in node_attr.items():
    #             if key == '__call__':
    #                 continue
    #
    #             if type_.startswith('List'):
    #                 g.edge(node_name, 'list')
    #             elif type_.startswith('Tuple'):
    #                 g.edge(node_name, 'Tuple')
    #             elif type_ not in func_dct.keys():
    #                 g.edge(node_name, type_)
    #             else:
    #                 c2.edge(node_name, type_)
    #     c2.attr(label = 'functions')
