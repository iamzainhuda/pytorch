import os

import torch._inductor.config as inductor_config

from torch.fx.graph_module import GraphModule

from .graph_drawer import FxGraphDrawer

__all__ = ["GraphTransformObserver"]


class GraphTransformObserver:
    __pass_count = 0

    def __init__(self, gm: GraphModule, passname: str):
        # If log_url_for_graph_transformation is None, we don't log anything
        if inductor_config.trace.log_url_for_graph_transformation is None:
            self.log_url = None
            return
        GraphTransformObserver.__pass_count += 1
        self.log_url = inductor_config.trace.log_url_for_graph_transformation
        self.gm = gm
        self.passname = passname

        self.input_dot_graph = FxGraphDrawer(
            self.gm,
            self.passname,
            ignore_getattr=True,
            ignore_parameters_and_buffers=True,
        ).get_dot_graph()

    @classmethod
    def get_current_pass_count(cls):
        return cls.__pass_count

    def __enter__(self):
        if self.log_url is None or self.gm is None:
            return self

        self.erased_nodes = set()
        self.created_nodes = set()
        self.gm._register_create_node_hook(self.on_node_creation)
        self.gm._register_erase_node_hook(self.on_node_erase)

        return self

    def __exit__(self, type, value, tb):
        if self.log_url is None or self.gm is None:
            return

        self.gm._unregister_create_node_hook(self.on_node_creation)
        self.gm._unregister_erase_node_hook(self.on_node_erase)

        if len(self.created_nodes) > 0 or len(self.erased_nodes) > 0:
            for e in self.input_dot_graph.get_node_list():
                if e.get_name() in self.erased_nodes:
                    e.obj_dict["attributes"]["fillcolor"] = "yellow"
                else:
                    e.obj_dict["attributes"]["fillcolor"] = "grey"
            self.input_dot_graph.write_svg(
                os.path.join(
                    self.log_url,
                    f"pass_{GraphTransformObserver.__pass_count}_{self.passname}_input_graph.svg",
                )
            )

            output_dot_graph = FxGraphDrawer(
                self.gm,
                self.passname,
                ignore_getattr=True,
                ignore_parameters_and_buffers=True,
            ).get_dot_graph()
            for e in output_dot_graph.get_node_list():
                if e.get_name() in self.created_nodes:
                    e.obj_dict["attributes"]["fillcolor"] = "yellow"
                else:
                    e.obj_dict["attributes"]["fillcolor"] = "grey"
            output_dot_graph.write_svg(
                os.path.join(
                    self.log_url,
                    f"pass_{GraphTransformObserver.__pass_count}_{self.passname}_output_graph.svg",
                )
            )

    def on_node_creation(self, node):
        self.created_nodes.add(node.name)

    def on_node_erase(self, node):
        self.erased_nodes.add(node.name)
