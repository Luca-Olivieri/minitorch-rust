use crate::core::autograd::grad_fn::BackwardSource;
use crate::core::dtype::Dtype;
use crate::core::storage::TensorStorage;

// TODO: why do I need a case for Leaf? isn't it enoough to have it None?

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
