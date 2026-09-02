macro_rules! impl_tensor_binary_ops {
    ($($trait:ident, $method:ident, $storage_fn:path, $grad_fn:ident, $grad_rule:ident);* $(;)?) => {
        $(
            impl_tensor_binary_op!($trait, $method, $storage_fn, $grad_fn, $grad_rule);
        )*
    };
    // Allow macro loop for non-differentiable ops
    ($($trait:ident, $method:ident, $storage_fn:path);* $(;)?) => {
        $(
            impl_tensor_binary_op!($trait, $method, $storage_fn);
        )*
    };
}

macro_rules! impl_tensor_binary_op {
    // 1. Trait implementation (Differentiable)
    ($trait:ident, $method:ident, $storage_fn:path, $grad_fn:ident, $grad_rule:ident) => {
        impl $trait for &GraphTensor {
            type Output = GraphTensor;
            fn $method(self, other: &GraphTensor) -> GraphTensor {
                apply_tensor_op(
                    |ops: &[&TensorStorage; 2]| $storage_fn(&[ops[0], ops[1]]),
                    Some(|operands: [GraphTensor; 2]| {
                        Box::new($grad_fn { operands, op: $grad_rule}) as Box<dyn GradFnTrait>
                    }),
                    &[self, other],
                )
            }
        }
    };
    // 2. Trait implementation (Non-differentiable)
    ($trait:ident, $method:ident, $storage_fn:path) => {
        impl $trait for &GraphTensor {
            type Output = GraphTensor;
            fn $method(self, other: &GraphTensor) -> GraphTensor {
                apply_tensor_op(
                    |ops: &[&TensorStorage; 2]| $storage_fn(&[ops[0], ops[1]]),
                    None::<fn([GraphTensor; 2]) -> Box<dyn GradFnTrait>>,
                    &[self, other],
                )
            }
        }
    };
    // 3. Method implementation (Differentiable)
    ($method:ident, $storage_fn:path, $grad_fn:ident, $grad_rule:ident) => {
        pub fn $method(&self, other: &GraphTensor) -> GraphTensor {
            apply_tensor_op(
                |ops: &[&TensorStorage; 2]| $storage_fn(&[ops[0], ops[1]]),
                Some(|operands: [GraphTensor; 2]| {
                    Box::new($grad_fn { operands, op: $grad_rule}) as Box<dyn GradFnTrait>
                }),
                &[self, other],
            )
        }
    };
    // 4. Method implementation (Non-differentiable)
    ($method:ident, $storage_fn:path) => {
        pub fn $method(&self, other: &GraphTensor) -> GraphTensor {
            apply_tensor_op(
                |ops: &[&TensorStorage; 2]| $storage_fn(&[ops[0], ops[1]]),
                None::<fn([GraphTensor; 2]) -> Box<dyn GradFnTrait>>,
                &[self, other],
            )
        }
    };
}

macro_rules! impl_tensor_unary_ops {
    ($($trait:ident, $method:ident, $storage_fn:path, $grad_fn:ident, $grad_rule:ident);* $(;)?) => {
        $(
            impl_tensor_unary_op!($trait, $method, $storage_fn, $grad_fn, $grad_rule);
        )*
    };
    // Allow macro loop for non-differentiable ops
    ($($trait:ident, $method:ident, $storage_fn:path);* $(;)?) => {
        $(
            impl_tensor_unary_op!($trait, $method, $storage_fn);
        )*
    };
}

macro_rules! impl_tensor_unary_op {
    // 1. Trait implementation (Differentiable)
    ($trait:ident, $method:ident, $storage_fn:path, $grad_fn:ident, $grad_rule:ident) => {
        impl $trait for &GraphTensor {
            type Output = GraphTensor;
            fn $method(self) -> GraphTensor {
                apply_tensor_op(
                    |ops: &[&TensorStorage; 1]| $storage_fn(&[ops[0]]),
                    Some(|operands: [GraphTensor; 1]| {
                        Box::new($grad_fn { operands, op: $grad_rule }) as Box<dyn GradFnTrait>
                    }),
                    &[self],
                )
            }
        }
    };
    // 2. Trait implementation (Non-differentiable)
    ($trait:ident, $method:ident, $storage_fn:path) => {
        impl $trait for &GraphTensor {
            type Output = GraphTensor;
            fn $method(self) -> GraphTensor {
                apply_tensor_op(
                    |ops: &[&TensorStorage; 1]| $storage_fn(&[ops[0]]),
                    None::<fn([GraphTensor; 1]) -> Box<dyn GradFnTrait>>,
                    &[self],
                )
            }
        }
    };
    // 3. Method implementation (Differentiable)
    ($method:ident, $storage_fn:path, $grad_fn:ident, $grad_rule:ident) => {
        pub fn $method(&self) -> GraphTensor {
            apply_tensor_op(
                |ops: &[&TensorStorage; 1]| $storage_fn(&[ops[0]]),
                Some(|operands: [GraphTensor; 1]| {
                    Box::new($grad_fn { operands, op: $grad_rule }) as Box<dyn GradFnTrait>
                }),
                &[self],
            )
        }
    };
    // 4. Method implementation (Non-differentiable)
    ($method:ident, $storage_fn:path) => {
        pub fn $method(&self) -> GraphTensor {
            apply_tensor_op(
                |ops: &[&TensorStorage; 1]| $storage_fn(&[ops[0]]),
                None::<fn([GraphTensor; 1]) -> Box<dyn GradFnTrait>>,
                &[self],
            )
        }
    };
}
