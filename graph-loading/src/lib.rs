use std::str::FromStr;

pub struct GraphData {
    pub nodes_edge_offset: Vec<u32>,
    pub nodes_edges: Vec<u32>,
}

pub fn parse_graph_data(data: &str) -> Option<GraphData> {
    let mut lines = data.lines();

    let header = lines.next()?;

    let mut parts = header.split_ascii_whitespace();

    let vertex_count = parts.next().expect("expected vertex count");
    let vertex_count =
        usize::from_str(vertex_count).expect("expected vertex count to be an integer");
    let edge_count = parts.next().expect("expected edge count");
    let edge_count = usize::from_str(edge_count).expect("expected edge count to be an integer");

    let mut nodes_edge_offset = Vec::with_capacity(vertex_count);
    let mut nodes_edges = Vec::with_capacity(edge_count);

    let mut current_offset = 0;

    for line in lines {
        let mut node_edge_count = 0;

        for part in line.split_ascii_whitespace() {
            let edge_ref = u32::from_str(part).unwrap_or_else(|_| {
                panic!("expected edge reference to be a number, found `{}`", part);
            });

            let edge_index = edge_ref - 1; // graph format is `1` indexed

            nodes_edges.push(edge_index);

            node_edge_count += 1;
        }

        nodes_edge_offset.push(current_offset);

        current_offset += node_edge_count;
    }

    Some(GraphData {
        nodes_edge_offset,
        nodes_edges,
    })
}
