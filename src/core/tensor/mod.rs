mod indexing;
pub mod init;
pub mod ops;

use std::fmt;
use std::rc::Rc;

use crate::core::autograd::grad_fn::GradFnTrait;
use crate::core::autograd::ops::shape::{BackwardCopyD, CopyDOp};
use crate::core::node::TensorNode;
use crate::core::storage::TensorStorage;
use crate::core::tensor::ops::math::apply_tensor_op;

/// Accessors shared by every tensor flavor (`GraphTensor`, `FreeTensor`).
///
/// This is the only dialect of tensor access that is exposed to library users:
/// the underlying graph node (`TensorNode`) and storage (`TensorStorage`) are
/// implementation details and are deliberately NOT surfaced here.
pub trait AbstractTensor {
    fn at(&self, md_idx: &[usize]) -> &f64;

    fn shape(&self) -> &Vec<usize>;

    fn numel(&self) -> usize;

    fn requires_grad(&self) -> bool;

    /// Read the whole tensor as a single scalar. Panics unless the tensor has
    /// exactly one element.
    fn item(&self) -> f64 {
        if self.numel() != 1 {
            panic!(
                "Cannot call item() on a non-singleton tensor (shape {:?}).",
                self.shape()
            );
        }
        *self.at(&vec![0; self.shape().len()])
    }

    /// New tensor backed by a deep copy of the data buffer. Graph-flavored
    /// tensors keep the graph edge (gradient flows through); free tensors stay
    /// disconnected from the graph.
    fn copy_d(&self) -> Self;

    /// True when no other handle shares this tensor's underlying node.
    ///
    /// Every operation that consumes a `GraphTensor` snapshots it into the graph
    /// (a clone of its node), so a graph tensor that has participated in any
    /// operation is shared and reports `false`. A `FreeTensor` owns its node by
    /// construction and is always solely referenced.
    ///
    /// This is the signal to use before state changes that rebind a tensor's node
    /// (e.g. freezing a module's parameters): once a tensor has been captured into
    /// a graph, rebinding it would orphan the gradients that graph computes for it.
    fn is_unique_ref(&self) -> bool;

    /// Render this tensor in a PyTorch-style, human-readable format, mirroring
    /// `TensorStorage`'s display. `label` names the concrete tensor flavor
    /// (`FreeTensor`, `GraphTensor`) on the opening line.
    fn fmt_tensor(&self, f: &mut fmt::Formatter<'_>, label: &str) -> fmt::Result {
        let continuation_indent = " ".repeat(label.len() + 1);
        writeln!(f, "{label}(shape={:?}, dtype=float,", self.shape())?;
        writeln!(f, "{continuation_indent}numel={},", self.numel())?;
        write!(f, "{continuation_indent}data=")?;

        if self.shape().is_empty() {
            write!(f, "{:.4}", self.at(&[]))?;
        } else {
            let mut curr_md_idx: Vec<usize> = Vec::new();
            print_tensor_recursive(
                self,
                f,
                0,
                &mut curr_md_idx,
                continuation_indent.len() + "data=".len(),
            )?;
        }

        write!(f, ")")
    }
}

fn print_tensor_recursive<T: AbstractTensor + ?Sized>(
    t: &T,
    f: &mut fmt::Formatter<'_>,
    dim_index: usize,
    curr_md_idx: &mut Vec<usize>,
    indent: usize,
) -> fmt::Result {
    let shape = t.shape();
    let dim_size = shape[dim_index];

    if dim_index == shape.len() - 1 {
        write!(f, "[")?;
        for i in 0..dim_size {
            curr_md_idx.push(i);
            let val = *t.at(curr_md_idx);
            curr_md_idx.pop();

            write!(f, "{:.4}", val)?;
            if i < dim_size - 1 {
                write!(f, ", ")?;
            }
        }
        write!(f, "]")?;
    } else {
        write!(f, "[")?;
        for i in 0..dim_size {
            if i > 0 {
                write!(f, ",")?;
                let newlines = shape.len() - dim_index - 1;
                for _ in 0..newlines {
                    writeln!(f)?;
                }
                for _ in 0..(indent + 1) {
                    write!(f, " ")?;
                }
            }
            curr_md_idx.push(i);
            print_tensor_recursive(t, f, dim_index + 1, curr_md_idx, indent + 1)?;
            curr_md_idx.pop();
        }
        write!(f, "]")?;
    }

    Ok(())
}

/// Crate-internal access to the computation-graph node. Not public: `TensorNode`
/// and `TensorStorage` are implementation details.
pub(crate) trait TensorNodeAccess: AbstractTensor {
    fn get_node(&self) -> &TensorNode;

    fn get_node_mut(&mut self) -> &mut TensorNode;
}

pub struct FreeTensor {
    node: Box<TensorNode>,
}

impl AbstractTensor for FreeTensor {
    fn at(&self, md_idx: &[usize]) -> &f64 {
        self.node.storage.at(md_idx)
    }

    fn shape(&self) -> &Vec<usize> {
        &self.node.storage.shape
    }

    fn numel(&self) -> usize {
        self.node.storage.numel
    }

    fn requires_grad(&self) -> bool {
        self.node.requires_grad
    }

    fn copy_d(&self) -> Self {
        let node = TensorNode {
            storage: TensorStorage::copy_d(&self.node.storage),
            requires_grad: self.node.requires_grad,
            grad_fn: None,
        };

        Self {
            node: Box::new(node),
        }
    }

    fn is_unique_ref(&self) -> bool {
        true
    }
}

impl TensorNodeAccess for FreeTensor {
    fn get_node(&self) -> &TensorNode {
        self.node.as_ref()
    }

    fn get_node_mut(&mut self) -> &mut TensorNode {
        self.node.as_mut()
    }
}

impl FreeTensor {
    pub fn new(shape: Vec<usize>, fill_value: f64, requires_grad: bool) -> Self {
        let node = TensorNode::new(shape, fill_value, requires_grad);

        Self {
            node: Box::new(node),
        }
    }

    /// Set the autograd flag in place. Only safe on the owned, uniquely held
    /// `FreeTensor`; `GraphTensor` nodes are immutable once shared in a graph.
    pub fn set_requires_grad(&mut self, requires_grad: bool) {
        self.node.requires_grad = requires_grad;
    }

    pub fn to_graph(self) -> GraphTensor {
        GraphTensor {
            node: Rc::from(self.node),
        }
    }
}

impl fmt::Display for FreeTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        AbstractTensor::fmt_tensor(self, f, "FreeTensor")
    }
}

impl fmt::Debug for FreeTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

pub struct GraphTensor {
    pub(super) node: Rc<TensorNode>,
}

impl GraphTensor {
    // turn this impl and the above one in a macro

    pub fn new(shape: Vec<usize>, fill_value: f64, requires_grad: bool) -> Self {
        let node = TensorNode::new(shape, fill_value, requires_grad);

        Self {
            node: Rc::new(node),
        }
    }

    /// Build a `GraphTensor` from an arbitrarily nested `Vec<f64>`.
    ///
    /// Both the shape and the underlying flat data are inferred from the
    /// structure of the input, e.g. `vec![1.0, 2.0]`, `vec![vec![1.0, 2.0]]`,
    /// `vec![vec![vec![1.0, 2.0]]]`, etc.
    ///
    /// Panics if the nested structure is ragged (children of the same parent
    /// have differing shapes).
    pub fn wrap<T: IntoNestedStorage>(data: T, requires_grad: bool) -> Self {
        let mut buffer: Vec<f64> = Vec::new();
        let shape = data.collect_into(&mut buffer);

        let storage = TensorStorage::from_buffer(shape, buffer);
        let node = TensorNode::from_storage(storage, requires_grad);

        Self {
            node: Rc::new(node),
        }
    }

    pub fn copy_s(&self) -> GraphTensor {
        Self {
            node: self.node.clone(),
        }
    }

    pub fn detach(&self, requires_grad: bool) -> GraphTensor {
        let node = TensorNode {
            storage: TensorStorage::copy_s(&self.node.storage),
            requires_grad,
            grad_fn: None,
        };

        Self {
            node: Rc::new(node),
        }
    }
}

/// Trait abstracting over arbitrarily nested `Vec<f64>` so that `GraphTensor::from_vec`
/// can be called with any nesting depth. Implementations recursively collect the flat
/// data into `buffer` and return the shape of the sub-structure they represent.
pub trait IntoNestedStorage {
    fn collect_into(self, buffer: &mut Vec<f64>) -> Vec<usize>;
}

impl IntoNestedStorage for f64 {
    fn collect_into(self, buffer: &mut Vec<f64>) -> Vec<usize> {
        buffer.push(self);
        Vec::new()
    }
}

impl<T: IntoNestedStorage> IntoNestedStorage for Vec<T> {
    fn collect_into(self, buffer: &mut Vec<f64>) -> Vec<usize> {
        let mut shape = vec![self.len()];

        let mut child_shape: Option<Vec<usize>> = None;
        for child in self {
            let cs = child.collect_into(buffer);
            match &child_shape {
                None => child_shape = Some(cs),
                Some(prev) => {
                    if *prev != cs {
                        panic!(
                            "Ragged nested Vec: children of the same parent have differing shapes."
                        );
                    }
                }
            }
        }

        if let Some(cs) = child_shape {
            shape.extend(cs);
        }

        shape
    }
}

impl AbstractTensor for GraphTensor {
    fn at(&self, md_idx: &[usize]) -> &f64 {
        self.node.storage.at(md_idx)
    }

    fn shape(&self) -> &Vec<usize> {
        &self.node.storage.shape
    }

    fn numel(&self) -> usize {
        self.node.storage.numel
    }

    fn requires_grad(&self) -> bool {
        self.node.requires_grad
    }

    fn copy_d(&self) -> Self {
        apply_tensor_op(
            |ops: &[&TensorStorage; 1]| TensorStorage::copy_d(ops[0]),
            Some(|operands: [GraphTensor; 1]| {
                Box::new(BackwardCopyD {
                    operands,
                    op: CopyDOp {},
                }) as Box<dyn GradFnTrait>
            }),
            &[self],
        )
    }

    fn is_unique_ref(&self) -> bool {
        Rc::strong_count(&self.node) == 1
    }
}

impl TensorNodeAccess for GraphTensor {
    fn get_node(&self) -> &TensorNode {
        self.node.as_ref()
    }

    fn get_node_mut(&mut self) -> &mut TensorNode {
        Rc::get_mut(&mut self.node)
            .expect("Failed to obtain mutable reference of GraphTensor with shared ownership.")
    }
}

impl fmt::Display for GraphTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        AbstractTensor::fmt_tensor(self, f, "GraphTensor")
    }
}

impl fmt::Debug for GraphTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

fn extract_requires_grad(operands: &[&GraphTensor]) -> bool {
    operands.iter().any(|t| t.get_node().requires_grad)
}
