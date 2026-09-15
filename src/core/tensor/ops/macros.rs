// Tensor-level op macros.
//
// Design (merged Stage 3/4): each op lives in exactly *one* dtype home
// (`impl<T: Numeric>`, `impl<T: Float>`, `impl GraphTensor<bool>`), chosen by what
// its backward math needs. Every backward closure builds an
// `NBackwardOp::<$op, N, T>` (no f64-only type aliases), so the same forward
// graph machinery serves every dtype.

macro_rules! impl_tensor_binary_ops {
    // Differentiable: `Numeric, Add, add, TensorStorage::add, AddOp;`
    ($($bound:ident, $trait:ident, $method:ident, $storage_fn:path, $grad_rule:ident);* $(;)?) => {
        $(
            impl_tensor_binary_op_trait!($bound, $trait, $method, $storage_fn, $grad_rule);
        )*
    };
    // Non-differentiable: `Numeric, Add, add, TensorStorage::add;`
    ($($bound:ident, $trait:ident, $method:ident, $storage_fn:path);* $(;)?) => {
        $(
            impl_tensor_binary_op_trait!($bound, $trait, $method, $storage_fn);
        )*
    };
}

macro_rules! impl_tensor_binary_op_trait {
    // 1. Trait implementation (Differentiable)
    ($bound:ident, $trait:ident, $method:ident, $storage_fn:path, $grad_rule:ident) => {
        impl<T: $bound> $trait for &GraphTensor<T> {
            type Output = GraphTensor<T>;
            fn $method(self, other: &GraphTensor<T>) -> GraphTensor<T> {
                apply_tensor_op(
                    |ops: &[&TensorStorage<T>; 2]| $storage_fn(&[ops[0], ops[1]]),
                    Some(|operands: [GraphTensor<T>; 2]| {
                        Box::new(NBackwardOp::<$grad_rule, 2, T> {
                            operands,
                            op: $grad_rule,
                        }) as Box<dyn GradFnTrait<T>>
                    }),
                    &[self, other],
                )
            }
        }
    };
    // 2. Trait implementation (Non-differentiable)
    ($bound:ident, $trait:ident, $method:ident, $storage_fn:path) => {
        impl<T: $bound> $trait for &GraphTensor<T> {
            type Output = GraphTensor<T>;
            fn $method(self, other: &GraphTensor<T>) -> GraphTensor<T> {
                apply_tensor_op(
                    |ops: &[&TensorStorage<T>; 2]| $storage_fn(&[ops[0], ops[1]]),
                    None::<fn([GraphTensor<T>; 2]) -> Box<dyn GradFnTrait<T>>>,
                    &[self, other],
                )
            }
        }
    };
}

macro_rules! impl_tensor_unary_ops {
    // Differentiable: `Float, Neg, neg, TensorStorage::neg, NegOp;`
    ($($bound:ident, $trait:ident, $method:ident, $storage_fn:path, $grad_rule:ident);* $(;)?) => {
        $(
            impl_tensor_unary_op_trait!($bound, $trait, $method, $storage_fn, $grad_rule);
        )*
    };
}

macro_rules! impl_tensor_unary_op_trait {
    // 1. Trait implementation (Differentiable)
    ($bound:ident, $trait:ident, $method:ident, $storage_fn:path, $grad_rule:ident) => {
        impl<T: $bound> $trait for &GraphTensor<T> {
            type Output = GraphTensor<T>;
            fn $method(self) -> GraphTensor<T> {
                apply_tensor_op(
                    |ops: &[&TensorStorage<T>; 1]| $storage_fn(&[ops[0]]),
                    Some(|operands: [GraphTensor<T>; 1]| {
                        Box::new(NBackwardOp::<$grad_rule, 1, T> {
                            operands,
                            op: $grad_rule,
                        }) as Box<dyn GradFnTrait<T>>
                    }),
                    &[self],
                )
            }
        }
    };
}

/// Method-only variants (e.g. compare ops, `maximum`): differentiable or not,
/// invoked *inside* an `impl<T: Bound> GraphTensor<T>` block. There is no
/// corresponding std trait, so no trait impl is generated.
macro_rules! impl_tensor_binary_method {
    ($method:ident, $storage_fn:path, $grad_rule:ident) => {
        pub fn $method(&self, other: &GraphTensor<T>) -> GraphTensor<T> {
            apply_tensor_op(
                |ops: &[&TensorStorage<T>; 2]| $storage_fn(&[ops[0], ops[1]]),
                Some(|operands: [GraphTensor<T>; 2]| {
                    Box::new(NBackwardOp::<$grad_rule, 2, T> {
                        operands,
                        op: $grad_rule,
                    }) as Box<dyn GradFnTrait<T>>
                }),
                &[self, other],
            )
        }
    };
    ($method:ident, $storage_fn:path) => {
        pub fn $method(&self, other: &GraphTensor<T>) -> GraphTensor<T> {
            apply_tensor_op(
                |ops: &[&TensorStorage<T>; 2]| $storage_fn(&[ops[0], ops[1]]),
                None::<fn([GraphTensor<T>; 2]) -> Box<dyn GradFnTrait<T>>>,
                &[self, other],
            )
        }
    };
}

macro_rules! impl_tensor_unary_method {
    ($method:ident, $storage_fn:path, $grad_rule:ident) => {
        pub fn $method(&self) -> GraphTensor<T> {
            apply_tensor_op(
                |ops: &[&TensorStorage<T>; 1]| $storage_fn(&[ops[0]]),
                Some(|operands: [GraphTensor<T>; 1]| {
                    Box::new(NBackwardOp::<$grad_rule, 1, T> {
                        operands,
                        op: $grad_rule,
                    }) as Box<dyn GradFnTrait<T>>
                }),
                &[self],
            )
        }
    };
}

macro_rules! impl_tensor_scalar_ops {
    // `Numeric, Add, add; Float, Sub, sub;` — scalar operand is `T`, i.e. the
    // tensor's own dtype (no silent cross-dtype coercion).
    ($($bound:ident, $trait:ident, $method:ident);* $(;)?) => {
        $(
            impl<T: $bound> $trait<T> for &GraphTensor<T> {
                type Output = GraphTensor<T>;
                fn $method(self, other: T) -> GraphTensor<T> {
                    let other_t = GraphTensor::<T>::new(vec![], other, false);
                    self.$method(&other_t)
                }
            }
        )*
    };
}