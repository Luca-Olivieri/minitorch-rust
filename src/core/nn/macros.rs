#[macro_export]
macro_rules! modules {
    (
        $model:ident {
            modules { $($mod_field:ident : $mod_ty:ty),* $(,)? },
            params { $($param_field:ident),* $(,)? }
        }
    ) => {
        pub struct $model {
            $(pub $mod_field: $mod_ty,)*
            $(pub $param_field: $crate::core::GraphTensor,)*
        }

        impl $crate::core::nn::module::Module for $model {
            fn for_each_own_param(
                &self,
                _f: &mut dyn FnMut(&str, &$crate::core::GraphTensor),
            ) {
                $(_f(stringify!($param_field), &self.$param_field);)*
            }

            fn for_each_own_param_mut(
                &mut self,
                _f: &mut dyn FnMut(&str, &mut $crate::core::GraphTensor),
            ) {
                $(_f(stringify!($param_field), &mut self.$param_field);)*
            }

            fn for_each_own_module(
                &self,
                _f: &mut dyn FnMut(&str, &dyn $crate::core::nn::module::Module),
            ) {
                $(_f(stringify!($mod_field), &self.$mod_field);)*
            }

            fn for_each_own_module_mut(
                &mut self,
                _f: &mut dyn FnMut(&str, &mut dyn $crate::core::nn::module::Module),
            ) {
                $(_f(stringify!($mod_field), &mut self.$mod_field);)*
            }

            fn module(&self, name: &str) -> Option<&dyn $crate::core::nn::module::Module> {
                match name {
                    $(stringify!($mod_field) => Some(&self.$mod_field),)*
                    _ => None,
                }
            }

            fn module_mut(
                &mut self,
                name: &str,
            ) -> Option<&mut dyn $crate::core::nn::module::Module> {
                match name {
                    $(stringify!($mod_field) => Some(&mut self.$mod_field),)*
                    _ => None,
                }
            }

            fn param(&self, name: &str) -> Option<&$crate::core::GraphTensor> {
                match name {
                    $(stringify!($param_field) => Some(&self.$param_field),)*
                    _ => None,
                }
            }

            fn param_mut(
                &mut self,
                name: &str,
            ) -> Option<&mut $crate::core::GraphTensor> {
                match name {
                    $(stringify!($param_field) => Some(&mut self.$param_field),)*
                    _ => None,
                }
            }
        }
    };

    (
        $model:ident {
            modules { $($mod_field:ident : $mod_ty:ty),* $(,)? }
        }
    ) => {
        modules! {
            $model {
                modules { $($mod_field: $mod_ty),* },
                params {}
            }
        }
    };

    (
        $model:ident {
            params { $($param_field:ident),* $(,)? }
        }
    ) => {
        modules! {
            $model {
                modules {},
                params { $($param_field),* }
            }
        }
    };

    ($model:ident {}) => {
        pub struct $model {}

        impl $crate::core::nn::module::Module for $model {}
    };
}
