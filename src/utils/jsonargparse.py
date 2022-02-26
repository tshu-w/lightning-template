# HACK: https://github.com/omni-us/jsonargparse/issues/129
from jsonargparse.actions import _ActionLink
from jsonargparse.namespace import split_key_leaf


def apply_instantiation_links(parser, cfg, source):
    if not hasattr(parser, "_links_group"):
        return
    for action in parser._links_group._group_actions:
        if action.apply_on != "instantiate" or source != action.source[0][1].dest:
            continue
        source_object = cfg[source]
        if action.source[0][0] == action.source[0][1].dest:
            value = action.compute_fn(source_object)
        else:
            attr = split_key_leaf(action.source[0][0])[1]
            if hasattr(source_object, attr):
                value = getattr(source_object, attr)
                if action.compute_fn is not None:
                    value = action.compute_fn(value)
                _ActionLink.set_target_value(action, value, cfg)


_ActionLink.apply_instantiation_links = apply_instantiation_links
