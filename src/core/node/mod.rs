use crate::core::autograd::grad_fn::BackwardSource;
use crate::core::dtype::Dtype;
use crate::core::storage::TensorStorage;

/// Autograd metadata attached to a tensor node.
///
/// `None` means that the tensor does not require gradients: it is frozen, was
/// produced by a non-differentiable boundary, or was created under `no_grad`.
/// `Some(Leaf)` means that the tensor requires gradients but is a leaf with no
/// backward operation of its own, such as a model parameter or a trainable leaf
/// input. `Some(Node(source))` means that the tensor is a differentiable
/// intermediate whose deferred backward rule is stored in `source`.
///
/// `Leaf` is distinct from `None` because `grad_fn` alone cannot distinguish a
/// trainable parameter from a frozen tensor: both have no backward operation,
/// but only the parameter participates in gradient accumulation.
#[derive(Debug)]
pub(crate) enum AutogradMeta {
    /// A differentiable leaf with no forward operation attached.
    Leaf,
    /// A differentiable operation with a deferred backward rule.
    Node(Box<BackwardSource>),
}

impl AutogradMeta {
    pub(crate) fn grad_fn(&self) -> Option<&BackwardSource> {
        match self {
            Self::Leaf => None,
            Self::Node(source) => Some(source),
        }
    }
}

#[derive(Debug)]
pub(crate) struct TensorNode<T: Dtype = f64> {
    pub(crate) storage: TensorStorage<T>,
    pub(crate) autograd: Option<AutogradMeta>,
    // Autograd metadata is absent for frozen tensors and no-grad results.
}

impl<T: Dtype> TensorNode<T> {
    pub(crate) fn requires_grad(&self) -> bool {
        self.autograd.is_some()
    }

    pub(crate) fn grad_fn(&self) -> Option<&BackwardSource> {
        self.autograd.as_ref().and_then(AutogradMeta::grad_fn)
    }

    pub(crate) fn new(shape: Vec<usize>, fill_value: T, requires_grad: bool) -> Self {
        Self {
            storage: TensorStorage::new(shape, fill_value),
            autograd: requires_grad.then_some(AutogradMeta::Leaf),
        }
    }

    pub(crate) fn from_storage(storage: TensorStorage<T>, requires_grad: bool) -> Self {
        Self {
            storage,
            autograd: requires_grad.then_some(AutogradMeta::Leaf),
        }
    }
}
