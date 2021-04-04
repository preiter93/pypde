# def merge_dicts_recursively(*dicts):
#     import itertools as it
#     """
#     Creates a dict whose keyset is the union of all the
#     input dictionaries.  The value for each key is based
#     on the first dict in the list with that key.
#     dicts later in the list have higher priority

#     When values are dictionaries, it is applied recursively
#     """
#     result = dict()
#     all_items = it.chain(*[d.items() for d in dicts])
#     for key, value in all_items:
#         if key in result and isinstance(result[key], dict) and isinstance(value, dict):
#             result[key] = merge_dicts_recursively(result[key], value)
#         else:
#             result[key] = value
#     return result

# def digest_config(obj,kwargs,caller_locals={}):
#     """
#     Sets init args and CONFIG values as local variables

#     The purpose of this function is to ensure that all
#     configuration of any object is inheritable, able to
#     be easily passed into instantiation, and is attached
#     as an attribute of the object.
#     """
#     # -- Assemble list of CONFIGs from all super classes
#     classes_in_hierarchy = [obj.__class__]
#     static_configs = []
#     # -- Check class for CONFIG 
#     while len(classes_in_hierarchy) > 0:
#         Class = classes_in_hierarchy.pop()
#         classes_in_hierarchy += Class.__bases__
#         if hasattr(Class, "CONFIG"):
#             static_configs.append(Class.CONFIG)
#     # -- Merge Configs to list
#     all_dicts = [kwargs]
#     all_dicts += static_configs
#     # -- Append Config entries to class obj
#     obj.__dict__ = merge_dicts_recursively(*reversed(all_dicts))