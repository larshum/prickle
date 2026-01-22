use super::*;
use crate::utils::pprint::*;

use std::fmt;

#[derive(Clone, Debug, PartialEq)]
pub enum NodeKind {
    Sequential, BlockLocal, ClusterLocal, InterBlock
}

impl NodeKind {
    fn from_sync_point_kind(kind: &SyncPointKind) -> NodeKind {
        match kind {
            SyncPointKind::BlockLocal => NodeKind::BlockLocal,
            SyncPointKind::BlockCluster => NodeKind::ClusterLocal,
            SyncPointKind::InterBlock => NodeKind::InterBlock,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum GraphNode {
    Root {label: Name, children: Vec<GraphNode>},
    Container {label: Name, kind: NodeKind, children: Vec<GraphNode>},
    ExtCall {label: Name},
    Leaf {label: Name},
}

#[derive(Clone, Debug, PartialEq)]
struct Edge { src: Name, dst: Name }

impl Edge {
    fn new(src: Name, dst: Name) -> Edge {
        Edge {src, dst}
    }
}

impl PrettyPrint for Edge {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        let (env, src) = self.src.pprint(env);
        let (env, dst) = self.dst.pprint(env);
        (env, format!("  {src} -> {dst};"))
    }
}

fn mk_edges(src: &Name, children: &Vec<GraphNode>) -> Vec<Edge> {
    children.iter()
        .map(|n| Edge::new(src.clone(), n.get_node_id()))
        .collect::<Vec<Edge>>()
}

#[derive(Clone, Debug)]
struct NodeTypes {
    pub root: Vec<Name>,
    pub inter_block: Vec<Name>,
    pub block_local: Vec<Name>,
    pub cluster_local: Vec<Name>,
    pub external: Vec<Name>
}

impl Default for NodeTypes {
    fn default() -> Self {
        NodeTypes {
            root: vec![],
            inter_block: vec![],
            block_local: vec![],
            cluster_local: vec![],
            external: vec![]
        }
    }
}

fn collect_node_types(mut nt: NodeTypes, node: &GraphNode) -> NodeTypes {
    match node {
        GraphNode::Root {label, children} => {
            nt.root.push(label.clone());
            children.iter().fold(nt, collect_node_types)
        },
        GraphNode::Container {label, kind, children} => {
            match kind {
                NodeKind::Sequential => (),
                NodeKind::BlockLocal => nt.block_local.push(label.clone()),
                NodeKind::ClusterLocal => nt.cluster_local.push(label.clone()),
                NodeKind::InterBlock => nt.inter_block.push(label.clone()),
            };
            children.iter().fold(nt, collect_node_types)
        },
        GraphNode::ExtCall {label} => {
            nt.external.push(label.clone());
            nt
        },
        GraphNode::Leaf {..} => nt
    }
}

impl NodeTypes {
    fn from(node: &GraphNode) -> NodeTypes {
        collect_node_types(NodeTypes::default(), node)
    }
}

impl GraphNode {
    fn get_node_id(&self) -> Name {
        match self {
            GraphNode::Root {label, ..} => label,
            GraphNode::Container {label, ..} => label,
            GraphNode::ExtCall {label} => label,
            GraphNode::Leaf {label} => label,
        }.clone()
    }

    fn print_nodes_by_type(
        &self,
        env: PrettyPrintEnv
    ) -> (PrettyPrintEnv, String, String, String, String, String) {
        let node_types = NodeTypes::from(&self);
        let (env, root) = pprint_iter(node_types.root.iter(), env, " ");
        let (env, ib) = pprint_iter(node_types.inter_block.iter(), env, " ");
        let (env, bl) = pprint_iter(node_types.block_local.iter(), env, " ");
        let (env, cl) = pprint_iter(node_types.cluster_local.iter(), env, " ");
        let (env, ext) = pprint_iter(node_types.external.iter(), env, " ");
        (env, root, ib, bl, cl, ext)
    }

    fn collect_edges(&self, mut edges: Vec<Edge>) -> Vec<Edge> {
        match self {
            GraphNode::Root {label, children} |
            GraphNode::Container {label, children, ..} => {
                edges.append(&mut mk_edges(label, children));
                children.iter().fold(edges, |edges, ch| ch.collect_edges(edges))
            },
            GraphNode::ExtCall {..} | GraphNode::Leaf {..} => edges
        }
    }

    fn print_edges(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        let edges = self.collect_edges(vec![]);
        pprint_iter(edges.iter(), env, "\n")
    }

    fn to_graphviz_str(&self) -> String {
        let env = PrettyPrintEnv::default();
        let (env, root, ib, bl, cl, ext) = self.print_nodes_by_type(env);
        let (_, edges_str) = self.print_edges(env);
        format!(
            "digraph G {{\n\
             {6}node [color=black, style=filled] {0}; // root node\n\
             {6}node [style=dashed] {1}; // inter-block parallel nodes\n\
             {6}node [style=bold] {2}; // block-local parallel nodes\n\
             {6}node [style=dotted] {3}; // cluster-local parallel nodes\n\
             {6}node [shape=diamond, style=\"\"] {4}; // external call nodes\n\
             {6}node [shape=circle, style=\"\"]; // default nodes\n\
             {5}\n\
            }}",
            root, ib, bl, cl, ext, edges_str, "  "
        )
    }
}

impl fmt::Display for GraphNode {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "{}", self.to_graphviz_str())
    }
}

fn make_graph_repr_helper(mut acc: Vec<GraphNode>, body: &[Stmt]) -> Vec<GraphNode> {
    if body.len() >= 3 {
        match &body[0..3] {
            [ Stmt::SyncPoint {kind: SyncPointKind::InterBlock, ..},
              Stmt::Expr {e: Expr::Call {id, ..} | Expr::PyCallback {id, ..}, ..},
              Stmt::SyncPoint {kind: SyncPointKind::InterBlock, ..} ] => {
                acc.push(GraphNode::ExtCall {label: id.clone()});
                return if body.len() > 3 {
                    make_graph_repr_helper(acc, &body[3..])
                } else {
                    acc
                };
            },
            _ => ()
        }
    }
    if body.len() >= 2 {
        match &body[0..2] {
            [ Stmt::For {var, body, ..}, Stmt::SyncPoint {kind, ..} ] => {
                let children = make_graph_repr_helper(vec![], &body[..]);
                let kind = NodeKind::from_sync_point_kind(kind);
                acc.push(GraphNode::Container {label: var.clone(), kind, children});
                return if body.len() > 2 {
                    make_graph_repr_helper(acc, &body[2..])
                } else {
                    acc
                };
            },
            _ => ()
        }
    }
    if body.len() > 0 {
        let kind = NodeKind::Sequential;
        let node = match &body[0] {
            Stmt::For {var, body, ..} => {
                let children = make_graph_repr_helper(vec![], &body[..]);
                GraphNode::Container {label: var.clone(), kind, children}
            },
            Stmt::While {body, ..} => {
                let label = Name::sym_str("while");
                let children = make_graph_repr_helper(vec![], &body[..]);
                GraphNode::Container {label, kind, children}
            },
            Stmt::If {thn, els, ..} => {
                let label = Name::sym_str("if");
                let children = make_graph_repr_helper(vec![], &thn[..]);
                let children = make_graph_repr_helper(children, &els[..]);
                GraphNode::Container {label, kind, children}
            },
            _ => GraphNode::Leaf {label: Name::sym_str("stmt")}
        };
        acc.push(node);
        make_graph_repr_helper(acc, &body[1..])
    } else {
        acc
    }
}

pub fn make_graph(body: &Vec<Stmt>) -> GraphNode {
    GraphNode::Root {
        label: Name::sym_str("root"),
        children: make_graph_repr_helper(vec![], body)
    }
}
