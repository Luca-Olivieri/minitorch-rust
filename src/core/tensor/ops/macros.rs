macro_rules! impl_tensor_binary_ops {
    ($($trait:ident, $method:ident, $storage_fn:path, $grad_fn:ident, $grad_rule:ident);* $(;)?) => {
        $(
            impl_tensor_binary_op!($trait, $method, $storage_fn, $grad_fn, $grad_rule);
        )*
    };
}

macro_rules! impl_tensor_binary_op {
    ($trait:ident, $method:ident, $storage_fn:path, $grad_fn:ident, $grad_rule:ident) => {
        impl $trait for &GraphTensor {
            type Output = GraphTensor;
            fn $method(self, other: &GraphTensor) -> GraphTensor {
                apply_tensor_op(
                    |ops: &[&TensorStorage; 2]| $storage_fn(&[ops[0], ops[1]]),
                    |operands: [GraphTensor; 2]| {
                        Box::new($grad_fn { operands, op: $grad_rule}) as Box<dyn GradFnTrait>
                    },
                    &[self, other],
                )
            }
        }
    };
    ($method:ident, $storage_fn:path, $grad_fn:ident, $grad_rule:ident) => {
        pub fn $method(&self, other: &GraphTensor) -> GraphTensor {
            apply_tensor_op(
                |ops: &[&TensorStorage; 2]| $storage_fn(&[ops[0], ops[1]]),
                |operands: [GraphTensor; 2]| {
                    Box::new($grad_fn { operands, op: $grad_rule}) as Box<dyn GradFnTrait>
                },
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
}

macro_rules! impl_tensor_unary_op {
    ($trait:ident, $method:ident, $storage_fn:path, $grad_fn:ident, $grad_rule:ident) => {
        impl $trait for &GraphTensor {
            type Output = GraphTensor;
            fn $method(self) -> GraphTensor {
                apply_tensor_op(
                    |ops: &[&TensorStorage; 1]| $storage_fn(&[ops[0]]),
                    |operands: [GraphTensor; 1]| {
                        Box::new($grad_fn { operands, op: $grad_rule }) as Box<dyn GradFnTrait>
                    },
                    &[self],
                )
            }
        }
    };
    ($method:ident, $storage_fn:path, $grad_fn:ident, $grad_rule:ident) => {
        pub fn $method(&self) -> GraphTensor {
            apply_tensor_op(
                |ops: &[&TensorStorage; 1]| $storage_fn(&[ops[0]]),
                |operands: [GraphTensor; 1]| {
                    Box::new($grad_fn { operands, op: $grad_rule }) as Box<dyn GradFnTrait>
                },
                &[self],
            )
        }
    };
}
