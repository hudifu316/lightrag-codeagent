import bisect
from tree_sitter import Node

def get_node_line_range(node: Node, line_offset_list):
    start_byte = node.start_byte
    end_byte = node.end_byte
    start_line = bisect.bisect_right(line_offset_list, start_byte) - 1
    end_line = bisect.bisect_right(line_offset_list, end_byte) - 1
    return start_line + 1, end_line + 1
